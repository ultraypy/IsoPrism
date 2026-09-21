from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import DataLoader, TensorDataset

from .data import choose_input_genes, load_dataset, normalize_log_cpm
from .metrics import evaluate_allocations, evaluate_spatial_moran, quality_gate
from .model import (
    IsoBudgetAttentionNet,
    IsoformAllocationNet,
    dirichlet_multinomial_allocation_loss,
    multinomial_allocation_loss,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(obs, holdout_column: str | None, holdout_value: str | None, seed: int):
    if holdout_column:
        if holdout_column not in obs.columns:
            raise ValueError(f"missing holdout column {holdout_column!r}")
        values = obs[holdout_column].astype(str).to_numpy()
        if holdout_value is None:
            holdout_value = sorted(set(values))[-1]
        valid = np.flatnonzero(values == str(holdout_value))
        train = np.flatnonzero(values != str(holdout_value))
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(obs))
        n_valid = max(1, int(round(0.2 * len(order))))
        valid, train = order[:n_valid], order[n_valid:]
    if len(train) < 10 or len(valid) < 10:
        raise ValueError(f"invalid split: train={len(train)}, validation={len(valid)}")
    return train, valid, holdout_value


def internal_tuning_split(indices: np.ndarray, seed: int, fraction: float = 0.1):
    """Split platform-A cells for early stopping without inspecting platform B."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(indices)
    n_tune = max(10, int(round(fraction * len(order))))
    if len(order) - n_tune < 10:
        raise ValueError(f"not enough training-domain cells for internal tuning: {len(order)}")
    return order[n_tune:], order[:n_tune]


def filter_targets(y: sparse.csr_matrix, train_idx: np.ndarray, min_umis: int, min_cells: int):
    train_y = y[train_idx]
    keep = (np.asarray(train_y.sum(axis=0)).ravel() >= min_umis) & (
        np.asarray((train_y > 0).sum(axis=0)).ravel() >= min_cells
    )
    return np.flatnonzero(keep)


def retain_multi_isoform_genes(target_idx: np.ndarray, isoform_genes: np.ndarray) -> np.ndarray:
    genes = isoform_genes[target_idx]
    _, inverse, counts = np.unique(genes, return_inverse=True, return_counts=True)
    return target_idx[counts[inverse] >= 2]


def parent_gene_abundance(
    matrix: sparse.csr_matrix, genes: np.ndarray, target_genes: np.ndarray
) -> sparse.csr_matrix:
    canonical = lambda value: re.sub(r"\.\d+$", "", str(value))
    lookup = {canonical(gene): i for i, gene in enumerate(genes)}
    rows, columns = [], []
    for column, gene in enumerate(target_genes):
        key = canonical(gene)
        if key in lookup:
            rows.append(lookup[key])
            columns.append(column)
    selector = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(genes), len(target_genes)),
    )
    return (matrix @ selector).tocsr()


def fit_capture_factors(
    short_parent: sparse.csr_matrix,
    long_isoforms: sparse.csr_matrix,
    isoform_gene_index: np.ndarray,
) -> np.ndarray:
    """Learn training-only gene-specific short/long capture efficiency."""
    short_bulk = np.asarray(short_parent.sum(axis=0)).ravel().astype(np.float64)
    isoform_bulk = np.asarray(long_isoforms.sum(axis=0)).ravel().astype(np.float64)
    long_bulk = np.zeros(short_parent.shape[1], dtype=np.float64)
    np.add.at(long_bulk, isoform_gene_index, isoform_bulk)
    global_scale = long_bulk.sum() / max(short_bulk.sum(), 1.0)
    prior_short = max(float(np.median(short_bulk[short_bulk > 0])) * 0.01, 1.0)
    factors = (long_bulk + prior_short * global_scale) / (short_bulk + prior_short)
    positive = factors[np.isfinite(factors) & (factors > 0)]
    if len(positive):
        low, high = np.quantile(positive, [0.01, 0.99])
        factors = np.clip(factors, low, high)
    factors[~np.isfinite(factors)] = global_scale
    return factors.astype(np.float32)


def build_model(args, n_genes, isoform_gene_index, isoform_ids):
    if args.model == "isobudget":
        return IsoBudgetAttentionNet(
            n_genes=n_genes,
            n_isoforms=len(isoform_ids),
            isoform_gene_index=torch.from_numpy(isoform_gene_index),
            isoform_ids=isoform_ids,
            d_model=args.d_model,
            n_programs=args.n_programs,
            n_heads=args.n_heads,
            dropout=args.dropout,
        )
    return IsoformAllocationNet(
        n_genes=n_genes,
        n_isoforms=len(isoform_ids),
        isoform_gene_index=torch.from_numpy(isoform_gene_index),
        hidden=args.hidden,
        dropout=args.dropout,
    )


def allocation_loss(model, model_name, x, counts, group_index):
    if model_name == "isobudget":
        probabilities, concentrations = model(x, return_concentration=True)
        loss = dirichlet_multinomial_allocation_loss(
            probabilities, counts, group_index, concentrations
        )
        return probabilities, loss
    probabilities = model(x)
    return probabilities, multinomial_allocation_loss(probabilities, counts)


def predict_in_batches(model, matrix: np.ndarray, device: torch.device, batch_size: int):
    batches = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(matrix), batch_size):
            end = min(start + batch_size, len(matrix))
            probabilities = model(torch.from_numpy(matrix[start:end]).to(device))
            batches.append(probabilities.float().cpu())
    return torch.cat(batches, dim=0).numpy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-column")
    parser.add_argument("--holdout-value")
    parser.add_argument(
        "--external-evaluation",
        action="store_true",
        help="Use an internal training-domain split for early stopping; inspect the holdout only once.",
    )
    parser.add_argument("--max-input-genes", type=int, default=2000)
    parser.add_argument("--max-target-isoforms", type=int, default=6000)
    parser.add_argument("--min-target-umis", type=int, default=10)
    parser.add_argument("--min-target-cells", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--model", choices=["mlp", "isobudget"], default="mlp")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-programs", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU")
    device = torch.device(args.device)
    dataset = load_dataset(args.data)
    train_idx, eval_idx, holdout_value = split_indices(
        dataset.obs, args.holdout_column, args.holdout_value, args.seed
    )
    if args.external_evaluation:
        fit_idx, tune_idx = internal_tuning_split(train_idx, args.seed)
    else:
        fit_idx, tune_idx = train_idx, eval_idx
    input_idx = choose_input_genes(dataset.x[train_idx], args.max_input_genes)
    target_idx = filter_targets(
        dataset.y, train_idx, args.min_target_umis, args.min_target_cells
    )
    target_idx = retain_multi_isoform_genes(target_idx, dataset.isoform_genes)
    if len(target_idx) > args.max_target_isoforms:
        totals = np.asarray(dataset.y[train_idx][:, target_idx].sum(axis=0)).ravel()
        target_idx = target_idx[np.argpartition(totals, -args.max_target_isoforms)[-args.max_target_isoforms:]]
        target_idx.sort()
        target_idx = retain_multi_isoform_genes(target_idx, dataset.isoform_genes)
    if len(target_idx) < 10:
        raise ValueError(f"only {len(target_idx)} target isoforms passed filtering")

    isoform_genes = dataset.isoform_genes[target_idx]
    target_gene_names, isoform_gene_index = np.unique(isoform_genes, return_inverse=True)
    x = normalize_log_cpm(dataset.x[:, input_idx])
    y = dataset.y[:, target_idx].toarray().astype(np.float32)
    x_train = torch.from_numpy(x[fit_idx])
    y_train = torch.from_numpy(y[fit_idx])
    x_valid = torch.from_numpy(x[tune_idx])
    y_valid = torch.from_numpy(y[tune_idx])
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True, pin_memory=True
    )

    structure_catalog = (
        dataset.isoform_structures
        if dataset.isoform_structures is not None
        else dataset.isoforms
    )
    target_structures = structure_catalog[target_idx].tolist()
    model = build_model(args, x.shape[1], isoform_gene_index, target_structures).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = None
    best_valid = float("inf")
    stale = 0
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                _, loss = allocation_loss(
                    model, args.model, xb, yb, model.isoform_gene_index
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            _, valid_loss_tensor = allocation_loss(
                model,
                args.model,
                x_valid.to(device),
                y_valid.to(device),
                model.isoform_gene_index,
            )
            valid_loss = float(valid_loss_tensor.cpu())
        row = {"epoch": epoch, "train_nll": float(np.mean(train_losses)), "valid_nll": valid_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        if valid_loss < best_valid - 1e-5:
            best_valid = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")

    refit_history = []
    if args.external_evaluation:
        # Select the epoch count using only platform A, then retrain from scratch
        # on every platform-A cell.  Platform B is still untouched at this point.
        best_epoch = min(history, key=lambda row: row["valid_nll"])["epoch"]
        set_seed(args.seed)
        all_train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])
            ),
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        model = build_model(args, x.shape[1], isoform_gene_index, target_structures).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        for epoch in range(1, best_epoch + 1):
            model.train()
            train_losses = []
            for xb, yb in all_train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    _, loss = allocation_loss(
                        model, args.model, xb, yb, model.isoform_gene_index
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.detach().cpu()))
            row = {
                "epoch": epoch,
                "train_nll": float(np.mean(train_losses)),
            }
            refit_history.append(row)
            print(json.dumps({"refit": row}), flush=True)
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
    model.load_state_dict(best_state)
    valid_prob = predict_in_batches(model, x[eval_idx], device, args.batch_size)
    train_parent = parent_gene_abundance(
        dataset.x[train_idx], dataset.genes, target_gene_names
    )
    valid_parent = parent_gene_abundance(
        dataset.x[eval_idx], dataset.genes, target_gene_names
    )
    capture_factors = fit_capture_factors(
        train_parent,
        dataset.y[train_idx][:, target_idx],
        isoform_gene_index,
    )
    valid_parent = valid_parent.multiply(capture_factors).tocsr()
    metrics = evaluate_allocations(
        dataset.y[eval_idx][:, target_idx],
        valid_prob,
        isoform_gene_index,
        valid_parent,
    )
    spatial_moran = None
    if {"spatial_x", "spatial_y"}.issubset(dataset.obs.columns):
        coordinates = dataset.obs.iloc[eval_idx][["spatial_x", "spatial_y"]].to_numpy()
        spatial_moran = evaluate_spatial_moran(
            dataset.y[eval_idx][:, target_idx],
            valid_prob,
            isoform_gene_index,
            coordinates,
            valid_parent,
        )
        metrics["spatial_moran_spearman"] = spatial_moran
    gate = quality_gate(metrics, spatial_moran)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_type": args.model,
        "state_dict": best_state,
        "input_genes": dataset.genes[input_idx].tolist(),
        "target_isoforms": dataset.isoforms[target_idx].tolist(),
        "isoform_structures": target_structures,
        "isoform_genes": isoform_genes.tolist(),
        "isoform_gene_index": isoform_gene_index.tolist(),
        "target_genes": target_gene_names.tolist(),
        "gene_capture_factors": capture_factors.tolist(),
        "hidden": args.hidden,
        "d_model": args.d_model,
        "n_programs": args.n_programs,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
    }
    torch.save(checkpoint, args.output / "model.pt")
    report = {
        "model_type": args.model,
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "data": str(args.data),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "train_cells": int(len(train_idx)),
        "internal_fit_cells": int(len(fit_idx)),
        "internal_tuning_cells": int(len(tune_idx)),
        "validation_cells": int(len(eval_idx)),
        "external_evaluation": args.external_evaluation,
        "holdout_column": args.holdout_column,
        "holdout_value": holdout_value,
        "n_input_genes": int(len(input_idx)),
        "n_target_isoforms": int(len(target_idx)),
        "n_target_genes": int(len(target_gene_names)),
        "best_validation_nll": best_valid,
        "evaluation_abundance_source": "training_calibrated_short_read_parent_gene_abundance_scaled_to_long_read_library_size",
        "elapsed_seconds": time.time() - started,
        "metrics": metrics,
        "quality_gate": gate,
        "history": history,
        "refit_history": refit_history,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"metrics": metrics, "quality_gate": gate}, ensure_ascii=False), flush=True)
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
