from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import DataLoader, TensorDataset

from .data import choose_input_genes, load_dataset, normalize_log_cpm
from .grouping import PseudoclusterBundle, build_pseudocluster_bundle
from .metrics import (
    evaluate_cluster_proportion_spearman,
    evaluate_cluster_usage_and_switches,
    evaluate_isoform_cell_spearman,
    evaluate_matrix_only_predictions,
    evaluate_pseudocell_isoform_spearman,
    evaluate_spatial_proportion_moran,
)
from .model import (
    HierarchicalIsoBudgetAttentionNet,
    dirichlet_multinomial_allocation_loss,
    shrunken_multinomial_allocation_loss,
)
from .train import (
    filter_targets,
    fit_capture_factors,
    internal_tuning_split,
    parent_gene_abundance,
    retain_multi_isoform_genes,
    set_seed,
    split_indices,
)


def training_prior(
    counts: np.ndarray, group_index: np.ndarray, smoothing: float = 0.5
) -> np.ndarray:
    bulk = counts.sum(axis=0).astype(np.float64)
    prior = np.empty_like(bulk)
    for group in range(int(group_index.max()) + 1):
        ix = np.flatnonzero(group_index == group)
        prior[ix] = (bulk[ix] + smoothing) / (
            bulk[ix].sum() + smoothing * len(ix)
        )
    return prior.astype(np.float32)


def preferred_strata(obs) -> tuple[np.ndarray | None, str | None]:
    for column in ("donor", "dataset", "section", "sample", "condition", "domain"):
        if column not in obs.columns:
            continue
        values = obs[column]
        valid = values.notna() & (values.astype(str) != "")
        if valid.mean() >= 0.8:
            return values.fillna("unknown").astype(str).to_numpy(), column
    return None, None


def make_bundle(
    raw_expression: sparse.csr_matrix,
    expression: np.ndarray,
    counts: np.ndarray,
    obs,
    seed: int,
    target_size: int,
) -> tuple[PseudoclusterBundle, str | None]:
    strata, column = preferred_strata(obs)
    bundle = build_pseudocluster_bundle(
        raw_expression,
        expression,
        counts,
        seed=seed,
        target_size=target_size,
        strata=strata,
    )
    return bundle, column


def build_model(
    n_genes: int,
    group_index: np.ndarray,
    isoform_ids: list[str],
    prior: np.ndarray,
    args,
) -> HierarchicalIsoBudgetAttentionNet:
    return HierarchicalIsoBudgetAttentionNet(
        n_genes=n_genes,
        n_isoforms=len(isoform_ids),
        isoform_gene_index=torch.from_numpy(group_index),
        isoform_ids=isoform_ids,
        prior_probabilities=prior,
        d_model=args.d_model,
        n_programs=args.n_programs,
        n_heads=args.n_heads,
        dropout=args.dropout,
        residual_gate_init=args.residual_gate_init,
    )


def cell_loader(
    expression: np.ndarray,
    counts: np.ndarray,
    bundle: PseudoclusterBundle,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(expression),
            torch.from_numpy(bundle.cell_context),
            torch.from_numpy(counts),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )


def cluster_loader(
    bundle: PseudoclusterBundle, batch_size: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(bundle.group_expression),
            torch.from_numpy(bundle.group_counts),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )


def run_epoch(
    model: HierarchicalIsoBudgetAttentionNet,
    cells: DataLoader,
    clusters: DataLoader,
    device: torch.device,
    args,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    cluster_iterator = iter(clusters)
    totals = {"loss": 0.0, "cell": 0.0, "cluster": 0.0, "denoise": 0.0, "gate": 0.0}
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for xb, cb, yb in cells:
            try:
                cluster_x, cluster_y = next(cluster_iterator)
            except StopIteration:
                cluster_iterator = iter(clusters)
                cluster_x, cluster_y = next(cluster_iterator)
            xb = xb.to(device, non_blocking=True)
            cb = cb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            cluster_x = cluster_x.to(device, non_blocking=True)
            cluster_y = cluster_y.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                cell_probability, _, concentrations, gates = model(
                    xb, cb, return_components=True
                )
                cluster_probability, _, _, _ = model(
                    cluster_x, cluster_x, return_components=True
                )
                cell_loss = shrunken_multinomial_allocation_loss(
                    cell_probability,
                    yb,
                    model.isoform_gene_index,
                    model.prior_logits.exp(),
                    prior_strength=args.prior_strength,
                )
                cluster_loss = dirichlet_multinomial_allocation_loss(
                    cluster_probability,
                    cluster_y,
                    model.isoform_gene_index,
                    concentrations,
                )
                if training and args.denoise_weight > 0:
                    denoise_loss = model.structure_consistency_loss(
                        mask_probability=args.structure_mask_probability
                    )
                else:
                    denoise_loss = torch.zeros((), device=device)
                gate_penalty = gates.mean()
                loss = (
                    args.cluster_loss_weight * cluster_loss
                    + args.cell_loss_weight * cell_loss
                    + args.denoise_weight * denoise_loss
                    + args.gate_penalty_weight * gate_penalty
                )
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            totals["loss"] += float(loss.detach().cpu())
            totals["cell"] += float(cell_loss.detach().cpu())
            totals["cluster"] += float(cluster_loss.detach().cpu())
            totals["denoise"] += float(denoise_loss.detach().cpu())
            totals["gate"] += float(gate_penalty.detach().cpu())
            batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def fit_model(
    model: HierarchicalIsoBudgetAttentionNet,
    train_cells: DataLoader,
    train_clusters: DataLoader,
    valid_cells: DataLoader,
    valid_clusters: DataLoader,
    device: torch.device,
    args,
) -> tuple[dict[str, torch.Tensor], list[dict], int]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = None
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_cells,
            train_clusters,
            device,
            args,
            optimizer,
            scaler,
        )
        valid_metrics = run_epoch(
            model,
            valid_cells,
            valid_clusters,
            device,
            args,
            optimizer=None,
            scaler=None,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_cell_loss": train_metrics["cell"],
            "train_cluster_loss": train_metrics["cluster"],
            "train_denoise_loss": train_metrics["denoise"],
            "train_gate": train_metrics["gate"],
            "valid_loss": valid_metrics["loss"],
            "valid_cell_loss": valid_metrics["cell"],
            "valid_cluster_loss": valid_metrics["cluster"],
            "valid_gate": valid_metrics["gate"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if valid_metrics["loss"] < best_score - 1e-5:
            best_score = valid_metrics["loss"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    return best_state, history, best_epoch


def refit_model(
    model: HierarchicalIsoBudgetAttentionNet,
    cells: DataLoader,
    clusters: DataLoader,
    epochs: int,
    device: torch.device,
    args,
) -> list[dict]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(1, epochs + 1):
        values = run_epoch(
            model, cells, clusters, device, args, optimizer, scaler
        )
        row = {"epoch": epoch, **{f"train_{key}": value for key, value in values.items()}}
        history.append(row)
        print(json.dumps({"refit": row}), flush=True)
    return history


def predict_in_batches(
    model: HierarchicalIsoBudgetAttentionNet,
    expression: np.ndarray,
    context: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(expression), batch_size):
            end = min(start + batch_size, len(expression))
            probability = model(
                torch.from_numpy(expression[start:end]).to(device),
                torch.from_numpy(context[start:end]).to(device),
            )
            output.append(probability.float().cpu())
    return torch.cat(output).numpy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-column", required=True)
    parser.add_argument("--holdout-value", required=True)
    parser.add_argument("--max-input-genes", type=int, default=2000)
    parser.add_argument("--max-target-isoforms", type=int, default=4000)
    parser.add_argument("--min-target-umis", type=int, default=10)
    parser.add_argument("--min-target-cells", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-programs", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--residual-gate-init", type=float, default=-1.5)
    parser.add_argument("--pseudocluster-size", type=int, default=50)
    parser.add_argument("--prior-strength", type=float, default=2.0)
    parser.add_argument("--cluster-loss-weight", type=float, default=1.0)
    parser.add_argument("--cell-loss-weight", type=float, default=0.25)
    parser.add_argument("--denoise-weight", type=float, default=0.05)
    parser.add_argument("--gate-penalty-weight", type=float, default=0.01)
    parser.add_argument("--structure-mask-probability", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU")
    device = torch.device(args.device)
    started = time.time()
    dataset = load_dataset(args.data)
    train_idx, eval_idx, holdout_value = split_indices(
        dataset.obs, args.holdout_column, args.holdout_value, args.seed
    )
    fit_idx, tune_idx = internal_tuning_split(train_idx, args.seed)
    input_idx = choose_input_genes(dataset.x[train_idx], args.max_input_genes)
    target_idx = filter_targets(
        dataset.y, train_idx, args.min_target_umis, args.min_target_cells
    )
    target_idx = retain_multi_isoform_genes(target_idx, dataset.isoform_genes)
    if len(target_idx) > args.max_target_isoforms:
        totals = np.asarray(dataset.y[train_idx][:, target_idx].sum(axis=0)).ravel()
        target_idx = target_idx[
            np.argpartition(totals, -args.max_target_isoforms)[-args.max_target_isoforms:]
        ]
        target_idx.sort()
        target_idx = retain_multi_isoform_genes(target_idx, dataset.isoform_genes)
    if len(target_idx) < 10:
        raise ValueError(f"only {len(target_idx)} target isoforms passed filtering")

    isoform_genes = dataset.isoform_genes[target_idx]
    target_genes, group_index = np.unique(isoform_genes, return_inverse=True)
    raw_input = dataset.x[:, input_idx].tocsr()
    expression = normalize_log_cpm(raw_input)
    counts = dataset.y[:, target_idx].toarray().astype(np.float32)
    structures = (
        dataset.isoform_structures
        if dataset.isoform_structures is not None
        else dataset.isoforms
    )[target_idx].tolist()

    fit_bundle, strata_column = make_bundle(
        raw_input[fit_idx],
        expression[fit_idx],
        counts[fit_idx],
        dataset.obs.iloc[fit_idx],
        args.seed,
        args.pseudocluster_size,
    )
    tune_bundle, _ = make_bundle(
        raw_input[tune_idx],
        expression[tune_idx],
        counts[tune_idx],
        dataset.obs.iloc[tune_idx],
        args.seed + 1,
        args.pseudocluster_size,
    )
    fit_prior = training_prior(counts[fit_idx], group_index)
    model = build_model(
        expression.shape[1], group_index, structures, fit_prior, args
    ).to(device)
    best_state, history, best_epoch = fit_model(
        model,
        cell_loader(expression[fit_idx], counts[fit_idx], fit_bundle, args.batch_size, True),
        cluster_loader(fit_bundle, args.batch_size, True),
        cell_loader(expression[tune_idx], counts[tune_idx], tune_bundle, args.batch_size, False),
        cluster_loader(tune_bundle, args.batch_size, False),
        device,
        args,
    )

    set_seed(args.seed)
    train_bundle, strata_column = make_bundle(
        raw_input[train_idx],
        expression[train_idx],
        counts[train_idx],
        dataset.obs.iloc[train_idx],
        args.seed,
        args.pseudocluster_size,
    )
    final_prior = training_prior(counts[train_idx], group_index)
    model = build_model(
        expression.shape[1], group_index, structures, final_prior, args
    ).to(device)
    refit_history = refit_model(
        model,
        cell_loader(expression[train_idx], counts[train_idx], train_bundle, args.batch_size, True),
        cluster_loader(train_bundle, args.batch_size, True),
        best_epoch,
        device,
        args,
    )
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }

    evaluation_bundle, evaluation_strata = make_bundle(
        raw_input[eval_idx],
        expression[eval_idx],
        counts[eval_idx],
        dataset.obs.iloc[eval_idx],
        args.seed + 2,
        args.pseudocluster_size,
    )
    probabilities = predict_in_batches(
        model,
        expression[eval_idx],
        evaluation_bundle.cell_context,
        device,
        args.batch_size,
    )
    train_parent = parent_gene_abundance(
        dataset.x[train_idx], dataset.genes, target_genes
    )
    evaluation_parent = parent_gene_abundance(
        dataset.x[eval_idx], dataset.genes, target_genes
    )
    capture_factors = fit_capture_factors(
        train_parent, dataset.y[train_idx][:, target_idx], group_index
    )
    predicted_parent = evaluation_parent.multiply(capture_factors).tocsr()
    evaluation_counts = dataset.y[eval_idx][:, target_idx].tocsr()
    metrics = evaluate_matrix_only_predictions(
        evaluation_counts, probabilities, group_index, predicted_parent
    )
    metrics.update(
        evaluate_isoform_cell_spearman(
            evaluation_counts, probabilities, group_index
        )
    )
    metrics.update(
        evaluate_pseudocell_isoform_spearman(
            evaluation_counts,
            probabilities,
            group_index,
            predicted_parent,
            evaluation_bundle.group_labels,
        )
    )
    metrics.update(
        evaluate_cluster_usage_and_switches(
            evaluation_counts,
            probabilities,
            group_index,
            predicted_parent,
            evaluation_bundle.coarse_labels,
        )
    )
    metrics.update(
        evaluate_cluster_proportion_spearman(
            evaluation_counts,
            probabilities,
            group_index,
            evaluation_bundle.coarse_labels,
        )
    )
    if {"spatial_x", "spatial_y"}.issubset(dataset.obs.columns):
        coordinates = dataset.obs.iloc[eval_idx][["spatial_x", "spatial_y"]].to_numpy()
        metrics["spatial_proportion_moran_spearman"] = evaluate_spatial_proportion_moran(
            evaluation_counts,
            probabilities,
            group_index,
            coordinates,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_type": "hierarchical_isobudget",
        "state_dict": best_state,
        "input_genes": dataset.genes[input_idx].tolist(),
        "target_isoforms": dataset.isoforms[target_idx].tolist(),
        "isoform_structures": structures,
        "isoform_genes": isoform_genes.tolist(),
        "isoform_gene_index": group_index.tolist(),
        "target_genes": target_genes.tolist(),
        "gene_capture_factors": capture_factors.tolist(),
        "prior_probabilities": final_prior.tolist(),
        "d_model": args.d_model,
        "n_programs": args.n_programs,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
        "residual_gate_init": args.residual_gate_init,
        "pseudocluster_size": args.pseudocluster_size,
        "seed": args.seed,
    }
    torch.save(checkpoint, args.output / "model.pt")
    report = {
        "model_type": "hierarchical_isobudget",
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "data": str(args.data),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "holdout_column": args.holdout_column,
        "holdout_value": holdout_value,
        "train_cells": int(len(train_idx)),
        "internal_fit_cells": int(len(fit_idx)),
        "internal_tuning_cells": int(len(tune_idx)),
        "evaluation_cells": int(len(eval_idx)),
        "n_input_genes": int(len(input_idx)),
        "n_target_isoforms": int(len(target_idx)),
        "n_target_genes": int(len(target_genes)),
        "training_pseudoclusters": int(len(train_bundle.group_expression)),
        "evaluation_pseudoclusters": int(len(evaluation_bundle.group_expression)),
        "median_training_pseudocluster_size": float(np.median(train_bundle.sizes)),
        "training_strata_column": strata_column,
        "evaluation_strata_column": evaluation_strata,
        "best_epoch": int(best_epoch),
        "hyperparameters": {
            "prior_strength": args.prior_strength,
            "cluster_loss_weight": args.cluster_loss_weight,
            "cell_loss_weight": args.cell_loss_weight,
            "denoise_weight": args.denoise_weight,
            "gate_penalty_weight": args.gate_penalty_weight,
            "structure_mask_probability": args.structure_mask_probability,
            "pseudocluster_size": args.pseudocluster_size,
        },
        "metrics": metrics,
        "history": history,
        "refit_history": refit_history,
        "elapsed_seconds": time.time() - started,
        "strict_evaluation": (
            "test long reads are used only as scoring truth; no test long-read "
            "library-size calibration"
        ),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "best_epoch": best_epoch,
                "metrics": metrics,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
