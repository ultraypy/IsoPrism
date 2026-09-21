"""Reproducible first-round benchmarking of published, isoform-adapted networks."""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_dataset, normalize_log_cpm
from .matrix_benchmark import TASKS, _ordered_indices, fit_pca, short_read_evaluation_groups
from .metrics import (evaluate_proportion_predictions, evaluate_cluster_proportion_spearman,
                      evaluate_spatial_proportion_moran)
from .published_models import METHODS, ROOT, UPSTREAM, build_published_model
from .train import internal_tuning_split, set_seed, split_indices


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(safe_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_tabular_source_transition(folder, old_config, new_config, output_root):
    """Reuse only unchanged tabular work after the chromosome-parser bug fix.

    Original source files and their hashes must be present. Compare the complete
    tabular model/attention and numerical training functions as syntax trees;
    refuse reuse for either chromosome-aware method or any scientific change.
    """
    if new_config["method"] not in {"TabResNet", "FT-Transformer"}:
        return False
    differing = {key for key in old_config.keys() | new_config.keys() if old_config.get(key) != new_config.get(key)}
    if not differing <= {"models_sha256", "runner_sha256"}:
        return False
    archive = output_root / "source_before_chromosome_fix"
    names = {
        "published_models.py": ("models_sha256", {"AllocationAdapter", "TabularAdapter", "sdpa_forward", "load_source"}),
        "published_benchmark.py": ("runner_sha256", {"load_task", "batches", "amp_context", "make_optimizer", "fit", "validate", "predict", "evaluate", "check_loss"}),
    }
    for filename, (hash_key, selected) in names.items():
        before, after = archive / filename, Path(__file__).with_name(filename)
        if not before.exists() or file_hash(before) != old_config[hash_key]:
            return False
        def definitions(path):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            return {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in selected}
        old_definitions, new_definitions = definitions(before), definitions(after)
        if set(old_definitions) != selected or old_definitions != new_definitions:
            return False
    write_json(folder / "source_transition_audit.json", {
        "reason": "Corrected chromosome parsing for BABEL/scButterfly only; tabular math and numerical trainer unchanged.",
        "original_configuration": old_config, "resumed_configuration": new_config,
        "archived_source_directory": archive, "syntax_tree_identity_verified": True,
        "test_performance_used": False,
    })
    write_json(folder / "config.json", new_config)
    return True


def verify_upstream():
    result = {}
    for repository, expected in UPSTREAM.items():
        folder = ROOT / "third_party" / repository
        actual = subprocess.check_output(["git", "-C", str(folder), "rev-parse", "HEAD"], text=True).strip()
        changes = subprocess.check_output(["git", "-C", str(folder), "diff", "--name-only"], text=True).strip()
        if actual != expected or changes:
            raise RuntimeError(f"upstream {repository} is not the pinned unmodified checkout")
        result[repository] = actual
    return result


def load_task(task, split_seed=2026, smoke_cells=None):
    spec = TASKS[task]
    dataset = load_dataset(ROOT / spec.data)
    train_idx, test_idx, _ = split_indices(dataset.obs, spec.holdout_column, spec.holdout_value, split_seed)
    # The existing canonical catalogue was selected using the source domain only.
    checkpoint_path = ROOT / spec.isobudget_checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    genes = list(map(str, checkpoint["input_genes"]))
    isoforms = list(map(str, checkpoint["target_isoforms"]))
    gi = _ordered_indices(dataset.genes, genes, "input gene")
    yi = _ordered_indices(dataset.isoforms, isoforms, "target isoform")
    if smoke_cells is not None:
        rng = np.random.default_rng(split_seed)
        train_idx = np.sort(rng.choice(train_idx, min(smoke_cells, len(train_idx)), replace=False))
        test_idx = np.sort(rng.choice(test_idx, min(128, len(test_idx)), replace=False))
    x_raw = dataset.x[train_idx][:, gi].toarray().astype(np.float32)
    train_x = normalize_log_cpm(dataset.x[train_idx][:, gi])
    test_x = normalize_log_cpm(dataset.x[test_idx][:, gi])
    train_y = dataset.y[train_idx][:, yi].toarray().astype(np.float32)
    # External targets are kept on CPU and never passed to fit()/predict().
    test_y = dataset.y[test_idx][:, yi].tocsr()
    _, test_pca, _ = fit_pca(train_x, test_x, 50, split_seed)
    clusters, _, grouping_metadata = short_read_evaluation_groups(test_pca, split_seed)
    coordinates = None
    if {"spatial_x", "spatial_y"}.issubset(dataset.obs.columns):
        coordinates = dataset.obs.iloc[test_idx][["spatial_x", "spatial_y"]].to_numpy()
    group_index = np.asarray(checkpoint["isoform_gene_index"], dtype=np.int64)
    structures = list(map(str, checkpoint.get("isoform_structures", checkpoint["target_isoforms"])))
    metadata = {
        "task": task, "split": f"{spec.holdout_column}={spec.holdout_value}",
        "data": spec.data, "train_cells": len(train_idx), "evaluation_cells": len(test_idx),
        "target_isoforms": len(isoforms), "input_genes": len(genes),
        "canonical_checkpoint": spec.isobudget_checkpoint,
        "canonical_checkpoint_sha256": file_hash(checkpoint_path),
        "input_genes_list": genes, "target_isoforms_list": isoforms,
        "isoform_gene_index": group_index.tolist(), "isoform_structures": structures,
        "train_cell_ids": dataset.obs.index[train_idx].astype(str).tolist(),
        "evaluation_cell_ids": dataset.obs.index[test_idx].astype(str).tolist(),
        "evaluation_grouping": grouping_metadata,
        "data_hashes": {name: file_hash(ROOT / spec.data / name) for name in
                        ("x_gene_counts.npz", "y_isoform_counts.npz", "genes.csv", "isoforms.csv", "obs.csv")},
    }
    return dict(x=train_x, y=train_y, raw_x=x_raw, test_x=test_x, test_y=test_y,
                groups=group_index, structures=structures, clusters=clusters,
                coordinates=coordinates, metadata=metadata)


def batches(n, size, rng=None):
    order = np.arange(n) if rng is None else rng.permutation(n)
    parts = list(np.array_split(order, max(1, math.ceil(n / size))))
    # BatchNorm cannot learn from singleton mini-batches.
    if len(parts) > 1 and len(parts[-1]) == 1:
        last = parts.pop()
        parts[-1] = np.concatenate([parts[-1], last])
    return parts


def amp_context(method, device):
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if method == "FT-Transformer" and device.type == "cuda" else contextlib.nullcontext())


def make_optimizer(model, method, lr):
    if method == "scButterfly":
        discriminator_parameters = {id(p) for p in model.discriminators.parameters()}
        parameters = [p for p in model.parameters() if id(p) not in discriminator_parameters]
        return torch.optim.Adam(parameters, lr=lr)
    if method == "BABEL":
        return torch.optim.Adam(model.parameters(), lr=lr)
    groups = model.network.make_parameter_groups() if method == "FT-Transformer" else model.parameters()
    return torch.optim.AdamW(groups, lr=lr, weight_decay=1e-5)


def check_loss(loss, name):
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite {name}")


@torch.no_grad()
def validate(model, x, y, method, batch_size, device):
    model.eval()
    numerator = 0.0
    denominator = float(y.sum())
    for ix in batches(len(x), batch_size):
        with amp_context(method, device):
            p = model(x[ix])
        numerator += float(-(y[ix] * p.float().clamp_min(1e-8).log()).sum())
    return numerator / max(denominator, 1)


def fit(data, method, lr, seed, device, epochs, patience, batch_size,
        pretrain_epochs, validation_indices=None, progress_path=None):
    set_seed(seed)
    model = build_published_model(method, data["x"].shape[1], data["groups"], data["structures"]).to(device)
    if validation_indices is None:
        fit_idx = np.arange(len(data["x"]))
        valid_idx = None
    else:
        fit_idx, valid_idx = validation_indices
    # Source-only tensors. A single coherent normalization is used for all methods.
    x = torch.as_tensor(data["x"][fit_idx], device=device)
    y = torch.as_tensor(data["y"][fit_idx], device=device)
    raw_x = torch.as_tensor(data["raw_x"][fit_idx], device=device)
    vx = torch.as_tensor(data["x"][valid_idx], device=device) if valid_idx is not None else None
    vy = torch.as_tensor(data["y"][valid_idx], device=device) if valid_idx is not None else None
    optimizer = make_optimizer(model, method, lr)
    discriminator_optimizer = (torch.optim.SGD(model.discriminators.parameters(), lr=5 * lr)
                               if method == "scButterfly" else None)
    rng = np.random.default_rng(seed)
    history = []
    start = time.perf_counter()
    if method == "scButterfly":
        for side in ("x", "y"):
            for epoch in range(pretrain_epochs):
                model.train()
                losses = []
                for ix in batches(len(x), batch_size, rng):
                    optimizer.zero_grad(set_to_none=True)
                    loss = model.pretrain_loss(x[ix], y[ix], side, 0.01 * min(1, (epoch + 1) / 5))
                    check_loss(loss, "VAE pretraining loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                event = {"stage": "pretrain_" + side, "epoch": epoch + 1,
                         "loss": float(np.mean(losses)), "elapsed_seconds": time.perf_counter() - start}
                history.append(event)
                print(json.dumps({"method": method, "lr": lr, **event}), flush=True)
                if progress_path:
                    write_json(progress_path, history)
        # Pretraining and joint fitting use separate optimizer histories, like upstream.
        optimizer = make_optimizer(model, method, lr)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        summed_loss, seen = 0.0, 0
        components = {}
        for ix in batches(len(x), batch_size, rng):
            if discriminator_optimizer is not None:
                discriminator_optimizer.zero_grad(set_to_none=True)
                loss_d = model.discriminator_loss(x[ix], y[ix])
                check_loss(loss_d, "discriminator loss")
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(model.discriminators.parameters(), 5, error_if_nonfinite=True)
                discriminator_optimizer.step()
                for p in model.discriminators.parameters():
                    p.requires_grad_(False)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(method, device):
                loss, details = model.loss(x[ix], y[ix], raw_x[ix], kl_weight=0.01 * min(1, epoch / 5))
            check_loss(loss, "training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            if discriminator_optimizer is not None:
                for p in model.discriminators.parameters():
                    p.requires_grad_(True)
            summed_loss += float(loss.detach()) * len(ix)
            seen += len(ix)
            for key, value in details.items():
                components[key] = components.get(key, 0) + float(value) * len(ix)
        value = validate(model, vx, vy, method, batch_size, device) if vx is not None else None
        event = {"stage": "tune" if vx is not None else "refit", "epoch": epoch,
                 "training_loss": summed_loss / seen, "validation_allocation_nll": value,
                 "components": {k: v / seen for k, v in components.items()},
                 "elapsed_seconds": time.perf_counter() - start}
        history.append(event)
        print(json.dumps({"method": method, "lr": lr, **event}), flush=True)
        if progress_path:
            write_json(progress_path, history)
        if value is not None:
            if not math.isfinite(value):
                raise FloatingPointError("non-finite validation allocation NLL")
            if value < best_loss - 1e-5:
                best_loss, best_epoch, stale = value, epoch, 0
            else:
                stale += 1
            if stale >= patience:
                break
    return model, {"best_validation_nll": best_loss if vx is not None else None,
                   "best_epoch": best_epoch if vx is not None else epochs,
                   "epochs_run": epoch, "history": history,
                   "seconds": time.perf_counter() - start,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "peak_allocated_gpu_mb": torch.cuda.max_memory_allocated(device) / 2**20
                   if device.type == "cuda" else None}


@torch.no_grad()
def predict(model, expression, method, batch_size, device):
    model.eval()
    output = []
    for ix in batches(len(expression), batch_size):
        x = torch.as_tensor(expression[ix], device=device)
        with amp_context(method, device):
            probabilities = model(x)
        output.append(probabilities.float().cpu().numpy())
    return np.concatenate(output)


def evaluate(data, probabilities):
    result = evaluate_proportion_predictions(data["test_y"], probabilities, data["groups"])
    result.update(evaluate_cluster_proportion_spearman(
        data["test_y"], probabilities, data["groups"], data["clusters"]))
    if data["coordinates"] is not None:
        result["spatial_proportion_moran_spearman"] = evaluate_spatial_proportion_moran(
            data["test_y"], probabilities, data["groups"], data["coordinates"])
    return result


def run_one(data, method, args, seed, device):
    folder = args.output_root / data["metadata"]["task"] / method / f"seed_{seed}"
    config = {"method": method, "seed": seed, "learning_rates": args.learning_rates,
              "max_epochs": args.epochs, "patience": args.patience,
              "batch_size": args.ft_batch_size if method == "FT-Transformer" else args.batch_size,
              "pretrain_epochs_per_modality": args.pretrain_epochs if method == "scButterfly" else 0,
              "smoke_cells": args.smoke_cells, "split_seed": args.split_seed,
              "input_context": "cell_only", "selection_metric": "source_validation_allocation_nll",
              "precision": "bfloat16_autocast" if method == "FT-Transformer" else "float32",
              "upstream": UPSTREAM, "runner_sha256": file_hash(Path(__file__)),
              "models_sha256": file_hash(Path(__file__).with_name("published_models.py")),
              "catalogue_sha256": data["metadata"]["canonical_checkpoint_sha256"]}
    config_path = folder / "config.json"
    if config_path.exists():
        old_config = json.loads(config_path.read_text(encoding="utf-8"))
        if old_config != config and not verify_tabular_source_transition(folder, old_config, config, args.output_root):
            raise RuntimeError(f"Refusing to overwrite a different experiment: {folder}")
        if (folder / "result.json").exists():
            print(f"Already complete: {folder}", flush=True)
            return json.loads((folder / "result.json").read_text(encoding="utf-8"))
    write_json(config_path, config)
    split = internal_tuning_split(np.arange(len(data["x"])), args.split_seed)
    write_json(folder / "source_tuning_split.json", {"fit_source_indices": split[0], "validation_source_indices": split[1]})
    candidates = []
    started = time.perf_counter()
    for lr in args.learning_rates:
        candidate_path = folder / f"tune_lr_{lr:g}.json"
        if candidate_path.exists():
            report = json.loads(candidate_path.read_text(encoding="utf-8"))
        else:
            print(f"Tuning {data['metadata']['task']} {method} seed={seed} lr={lr}", flush=True)
            progress_path = folder / f"progress_tune_lr_{lr:g}.json"
            if progress_path.exists():
                write_json(folder / f"interrupted_tune_lr_{lr:g}_{time.strftime('%Y%m%d_%H%M%S')}.json",
                           json.loads(progress_path.read_text(encoding="utf-8")))
            model, report = fit(
                data, method, lr, seed, device, args.epochs, args.patience,
                config["batch_size"], config["pretrain_epochs_per_modality"], split,
                progress_path)
            del model
            torch.cuda.empty_cache()
            report["learning_rate"] = lr
            write_json(candidate_path, report)
        candidates.append(report)
    best = min(candidates, key=lambda r: r["best_validation_nll"])
    write_json(folder / "selection.json", {"learning_rate": best["learning_rate"],
               "refit_epochs": best["best_epoch"], "source_validation_nll": best["best_validation_nll"],
               "validation_cells": len(split[1]), "test_labels_used": False})
    checkpoint_path = folder / "model.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_published_model(method, data["x"].shape[1], data["groups"], data["structures"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        refit_report = checkpoint["refit"]
    else:
        print(f"Refitting {data['metadata']['task']} {method} for {best['best_epoch']} epochs", flush=True)
        model, refit_report = fit(
            data, method, best["learning_rate"], seed, device, best["best_epoch"], args.patience,
            config["batch_size"], config["pretrain_epochs_per_modality"], None, folder / "progress_refit.json")
        checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                      "config": config, "metadata": data["metadata"], "selection": best,
                      "refit": refit_report}
        torch.save(checkpoint, checkpoint_path.with_suffix(".pt.tmp"))
        checkpoint_path.with_suffix(".pt.tmp").replace(checkpoint_path)
    probabilities = predict(model, data["test_x"], method, config["batch_size"], device)
    # Inference signature has no target counts or library-size argument.
    metrics = evaluate(data, probabilities)
    np.savez_compressed(folder / "predicted_proportions.npz", probabilities=probabilities,
                        cell_ids=np.asarray(data["metadata"]["evaluation_cell_ids"]),
                        isoforms=np.asarray(data["metadata"]["target_isoforms_list"]),
                        cluster_labels=data["clusters"])
    result = {"task": data["metadata"]["task"], "method": method + " (isoform-adapted)",
              "seed": seed, "configuration": config, "metrics": metrics,
              "metadata": {k: data["metadata"][k] for k in
                           ("split", "train_cells", "evaluation_cells", "target_isoforms", "input_genes")},
              "selected_learning_rate": best["learning_rate"], "selected_epochs": best["best_epoch"],
              "source_validation_nll": best["best_validation_nll"], "refit": refit_report,
              "elapsed_seconds_this_invocation": time.perf_counter() - started,
              "checkpoint": str(checkpoint_path), "checkpoint_sha256": file_hash(checkpoint_path),
              "device": str(device), "gpu": torch.cuda.get_device_name(device),
              "torch_version": torch.__version__, "python_version": platform.python_version()}
    write_json(folder / "result.json", result)
    print(json.dumps(safe_json({"COMPLETED": result["task"], "method": result["method"],
                               "metrics": metrics}), ensure_ascii=False), flush=True)
    del model, probabilities
    torch.cuda.empty_cache()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[3e-4, 1e-3])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ft-batch-size", type=int, default=8)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--smoke-cells", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/published_benchmark_v1")
    args = parser.parse_args(argv)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("This training entrypoint requires an available CUDA GPU")
    if args.smoke_cells is not None and "smoke" not in str(args.output_root).lower():
        raise ValueError("smoke runs must use a separate output path containing 'smoke'")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    verify_upstream()
    run_manifest = {"args": vars(args), "upstream": UPSTREAM, "command": sys.argv,
                    "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
                    "model_sha256": file_hash(Path(__file__).with_name("published_models.py"))}
    # Each invocation has its own manifest; never clobber another command's audit trail.
    write_json(args.output_root / f"invocation_{time.strftime('%Y%m%d_%H%M%S')}.json", run_manifest)
    results = []
    for task in args.tasks:
        print(f"Loading canonical task {task}", flush=True)
        data = load_task(task, args.split_seed, args.smoke_cells)
        metadata_path = args.output_root / task / "data_manifest.json"
        if metadata_path.exists():
            if json.loads(metadata_path.read_text(encoding="utf-8")) != safe_json(data["metadata"]):
                raise RuntimeError(f"Canonical task changed: {task}")
        else:
            write_json(metadata_path, data["metadata"])
        for method in args.methods:
            for seed in args.seeds:
                results.append(run_one(data, method, args, seed, device))
        del data
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
