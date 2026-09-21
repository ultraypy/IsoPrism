"""Isolated GPU pilot for competitive allocation; frozen benchmark data are read-only."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .competitive_model import CompetitiveAllocationNet, multinomial_nll_terms, unsmoothed_cluster_targets
from .conservative_model import macro_jsd, macro_cross_entropy
from .published_benchmark import file_hash, safe_json
from .train import set_seed
from .unified_data import load_view
from .unified_metrics import evaluate

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "reports/competitive_allocation_protocol_20260909.md"
LABELS = {"cross_donor": "Cross-donor", "cross_dataset": "Cross-dataset", "cross_disease": "Normal-to-tumor transfer",
          "cross_platform": "Cross-platform", "spatial": "Spatial transfer"}
VARIANTS = {"IsoPrism": (4, "structure"), "Competitive": (4, "structure"), "Competitive-TA": (4, "structure"),
            "Single-step": (1, "structure"), "Identity-query": (4, "identity")}


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".record-", suffix=".pending", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(safe_json(payload), stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.1 * (attempt + 1))


def tensor(values, device):
    return torch.tensor(np.asarray(values), dtype=torch.float32, device=device)


def batches(n, size):
    for start in range(0, n, size):
        yield np.arange(start, min(n, start + size))


def configuration(view, variant, args):
    steps, query = VARIANTS[variant]
    cfg = {"n_input": len(view["input_genes"]), "groups": view["groups"].tolist(),
            "structures": view["structures"], "d_model": args.d_model, "n_programs": 8,
            "n_heads": 4, "steps": steps, "dropout": .1, "query_mode": query, "horizon": 1.0}
    if variant in {"IsoPrism", "Competitive-TA"}:
        cfg["terminal_features"] = view["terminal_features"]
    return cfg


def source_view(folder, smoke):
    view = load_view(folder)
    if smoke:
        # Mechanism/safety smoke only. Keep the full output catalogue, restrict
        # source rows and remap their PREEXISTING SR-only context labels.
        for name in ["fit", "val", "cal", "train"]:
            if f"{name}_x" not in view:
                continue
            n = min(96, len(view[f"{name}_x"]))
            for suffix in ["x", "y", "raw", "ids", "context"]:
                if f"{name}_{suffix}" in view:
                    view[f"{name}_{suffix}"] = view[f"{name}_{suffix}"][:n]
            old = view[f"{name}_context_labels"][:n]
            values, remapped = np.unique(old, return_inverse=True)
            view[f"{name}_context_labels"] = remapped
            view[f"{name}_context_means"] = view[f"{name}_context_means"][values]
    return view


@torch.no_grad()
def validate(model, view, args):
    model.eval()
    total, count = 0., 0
    groups = torch.as_tensor(view["groups"], device=args.device)
    for ix in batches(len(view["val_x"]), args.microbatch):
        p = model(tensor(view["val_x"][ix], args.device), tensor(view["val_context"][ix], args.device))
        numerator, denominator = macro_jsd(p, tensor(view["val_y"][ix], args.device), groups)
        total += numerator.item()
        count += denominator.item()
    if not count:
        raise ValueError("No eligible source-validation cell-gene pairs")
    return total / count


def fit(view, variant, args, seed, lr, folder, tag, epochs=None):
    set_seed(seed)
    name = "fit" if epochs is None else "train"
    cfg = configuration(view, variant, args)
    model = CompetitiveAllocationNet(**cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    cy, cm = unsmoothed_cluster_targets(view[f"{name}_y"], view[f"{name}_context_labels"], view["groups"])
    supported = np.flatnonzero(cm.any(1))
    groups = model.isoform_gene_index
    rng = np.random.default_rng(seed)
    history, best, best_epoch, stale, best_state = [], float("inf"), 0, 0, None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(args.device)
    for epoch in range(1, (args.epochs if epochs is None else epochs) + 1):
        model.train()
        order = rng.permutation(len(view[f"{name}_x"]))
        loss_sum, updates = 0., 0
        for offset in range(0, len(order), args.batch_size):
            selected = order[offset:offset + args.batch_size]
            observed_total = float(np.asarray(view[f"{name}_y"][selected]).sum(dtype=np.float64))
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.
            for begin in range(0, len(selected), args.microbatch):
                ix = selected[begin:begin + args.microbatch]
                xb, cb, yb = [tensor(view[f"{name}_{suffix}"][ix], args.device) for suffix in ["x", "context", "y"]]
                p = model(xb, cb)
                numerator, _ = multinomial_nll_terms(p, yb)
                loss = numerator / max(observed_total, 1.)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite multinomial objective")
                loss.backward()
                step_loss += loss.item()
                del xb, cb, yb, p, numerator, loss
            if len(supported) and args.cluster_weight:
                cix = rng.choice(supported, min(8, len(supported)), replace=False)
                cx = tensor(view[f"{name}_context_means"][cix], args.device)
                cp = model(cx, cx)
                cluster_loss = args.cluster_weight * macro_cross_entropy(cp, tensor(cy[cix], args.device),
                    torch.tensor(cm[cix], device=args.device), groups)
                if not torch.isfinite(cluster_loss):
                    raise FloatingPointError("Non-finite cluster objective")
                cluster_loss.backward()
                step_loss += cluster_loss.item()
                del cx, cp, cluster_loss
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite gradient norm")
            optimizer.step()
            loss_sum += step_loss
            updates += 1
        score = validate(model, view, args) if epochs is None else None
        if score is not None:
            if score < best - 1e-7:
                best, best_epoch, stale = score, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        event = {"epoch": epoch, "loss": loss_sum / max(updates, 1), "validation_jsd": score,
                 "seconds": time.perf_counter() - started, "gpu_peak_mb": torch.cuda.max_memory_allocated(args.device) / 1024**2}
        history.append(event)
        save_json(folder / f"progress_{tag}.json", history)
        print(json.dumps({"variant": variant, "stage": tag, **event}), flush=True)
        if epochs is None and stale >= args.patience:
            break
    if epochs is None:
        if best_state is None:
            raise ValueError("No finite validation checkpoint")
        model.load_state_dict(best_state)
    else:
        best_epoch = epoch
    record = {"learning_rate": lr, "selected_epoch": best_epoch,
              "validation_jsd": best if epochs is None else None, "history": history,
              "parameters": sum(p.numel() for p in model.parameters()),
              "peak_gpu_mb": torch.cuda.max_memory_allocated(args.device) / 1024**2,
              "trained_to_budget": epoch == args.epochs if epochs is None else None,
              "selected_at_budget": best_epoch == args.epochs if epochs is None else None,
              "seconds": time.perf_counter() - started}
    save_json(folder / f"{tag}.json", record)
    torch.save({"model_config": cfg, "model_state": model.state_dict()}, folder / f"{tag}.pt")
    return model, record


@torch.no_grad()
def mechanism_checks(model, view, args, name):
    model.eval()
    x = tensor(view[f"{name}_x"][:16], args.device)
    context = tensor(view[f"{name}_context"][:16], args.device)
    full, states = model(x, context, return_trajectory=True)
    one = model(x, context, steps=1)
    frozen = model(x, context, frozen_state=True)
    max_sum_error = 0.
    for p in states:
        sums = p.new_zeros((len(p), model.n_target_genes))
        sums.scatter_add_(1, model.isoform_gene_index[None].expand(len(p), -1), p)
        max_sum_error = max(max_sum_error, float((sums - 1).abs().max()))
        assert torch.isfinite(p).all() and (p >= 0).all()
    equivalent_error = float((one - frozen).abs().max())
    assert max_sum_error < 1e-5 and equivalent_error < 1e-5
    return {"max_gene_sum_error_all_iterations": max_sum_error,
            "one_step_vs_frozen_state_max_abs": equivalent_error,
            "state_dependent_vs_frozen_state_max_abs": float((full - frozen).abs().max()),
            "iteration_mean_abs_changes": [float((states[i + 1] - states[i]).abs().mean()) for i in range(len(states) - 1)],
            "no_prior_or_calibration_parameters": not any(any(word in key for word in ["prior", "calibration", "reference"]) for key in model.state_dict())}


def run_one(task, variant, args):
    folder = args.output_root / task / variant / f"seed_{args.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "result.json").exists():
        return json.loads((folder / "result.json").read_text(encoding="utf-8"))
    prepared = args.prepared_root / "prepared" / task
    inner = source_view(prepared / "inner", args.smoke)
    if variant in {"IsoPrism", "Competitive-TA"}:
        inner["terminal_features"] = json.loads((args.output_root / "structural_features" / task / "inner.json").read_text())["features"]
    records = []
    for lr in args.learning_rates:
        model, record = fit(inner, variant, args, args.seed, lr, folder, f"tune_{lr:g}")
        record["mechanism_checks"] = mechanism_checks(model, inner, args, "fit")
        records.append(record)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    best = min(records, key=lambda row: row["validation_jsd"])
    selection = {"tuning": records, "best": best, "criterion": "source validation cell-gene base-2 JSD", "calibration": "none"}
    save_json(folder / "selection.json", selection)
    if args.smoke:
        result = {"task": task, "variant": variant, "seed": args.seed, "smoke": True,
                  "selection": selection, "note": "No external truth read or scientific performance reported."}
        save_json(folder / "result.json", result)
        return result
    del inner
    gc.collect()
    final = source_view(prepared / "final", False)
    if variant in {"IsoPrism", "Competitive-TA"}:
        final["terminal_features"] = json.loads((args.output_root / "structural_features" / task / "final.json").read_text())["features"]
    model, refit = fit(final, variant, args, args.seed, best["learning_rate"], folder, "refit", epochs=best["selected_epoch"])
    catalogue = {k: final[k] for k in ["input_genes", "isoforms", "parents", "structures"]}
    catalogue["groups"] = final["groups"].tolist()
    checkpoint = {"model_config": configuration(final, variant, args), "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                  "catalogue": catalogue, "preprocessing": {"normalization": "log1p CPM over matched selected input genes",
                    "context_target_size": 50, "context_seed": 2026, "raw_counts_required": True},
                  "variant": variant, "selection": {"learning_rate": best["learning_rate"], "epochs": best["selected_epoch"]}}
    checkpoint_path = folder / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    diagnostics = mechanism_checks(model, final, args, "train")
    model.eval()
    shape = (len(final["test_x"]), len(final["groups"]))
    probabilities = np.lib.format.open_memmap(folder / "probabilities.npy", mode="w+", dtype=np.float32, shape=shape)
    with torch.no_grad():
        for ix in batches(shape[0], args.microbatch):
            probabilities[ix] = model(tensor(final["test_x"][ix], args.device), tensor(final["test_context"][ix], args.device)).cpu().numpy()
        cluster_probs = []
        for ix in batches(len(final["test_context_means"]), args.microbatch):
            context = tensor(final["test_context_means"][ix], args.device)
            cluster_probs.append(model(context, context).cpu().numpy())
    probabilities.flush()
    np.save(folder / "context_probabilities.npy", np.concatenate(cluster_probs))
    np.savez_compressed(folder / "prediction_index.npz", cell_ids=final["test_ids"], isoforms=np.asarray(final["isoforms"]),
                        evaluation_clusters=final["evaluation_clusters"], context_labels=final["test_context_labels"])
    del model, cluster_probs
    gc.collect()
    torch.cuda.empty_cache()
    # Checkpoint reconstruction and model inference receive no target LR matrix.
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    restored = CompetitiveAllocationNet(**saved["model_config"]).to(args.device)
    restored.load_state_dict(saved["model_state"])
    restored.eval()
    with torch.no_grad():
        p = restored(tensor(final["test_x"][:16], args.device), tensor(final["test_context"][:16], args.device)).cpu().numpy()
    error = float(np.max(np.abs(p - probabilities[:16])))
    assert error < 1e-5
    del restored, saved
    gc.collect()
    torch.cuda.empty_cache()
    # Only the evaluator opens held-out long-read counts.
    metrics = evaluate(final, probabilities)
    repeated = evaluate(final, probabilities)
    assert safe_json(metrics) == safe_json(repeated), "Metric recomputation mismatch"
    result = {"task": task, "variant": variant, "seed": args.seed, "smoke": False,
              "selection": selection, "refit": refit, "metrics": metrics, "mechanism_checks": diagnostics,
              "checkpoint_roundtrip_max_abs": error, "checkpoint_sha256": file_hash(checkpoint_path),
              "prediction_sha256": file_hash(folder / "probabilities.npy"),
              "prediction_index_sha256": file_hash(folder / "prediction_index.npz"),
              "context_prediction_sha256": file_hash(folder / "context_probabilities.npy"),
              "manifest_sha256": file_hash(prepared / "manifest.json"), "metrics_recomputed_identically": True}
    save_json(folder / "result.json", result)
    print(json.dumps({"completed": f"{task}/{variant}", "metrics": {k: safe_json(v) for k, v in metrics.items() if not isinstance(v, list)}}), flush=True)
    del final, probabilities
    gc.collect()
    return result


def report(args, rows, failures):
    save_json(args.output_root / "summary.json", {"pilot": True, "smoke": args.smoke, "runs": rows, "failures": failures})
    if args.smoke:
        return
    fields = ["confident_dominant_isoform_gene_macro_accuracy", "cell_gene_1_minus_jsd",
              "cluster_allocation_spearman_median", "spatial_proportion_moran_spearman", "cluster_contrast_mae"]
    lines = ["# Competitive allocation: initial pilot", "", "Single seed, source-only selection, no Mean prior or calibration. Preliminary results, not a complete benchmark.", "",
             "| Setting | Variant | Dominant accuracy ↑ | Cell-gene 1−JSD ↑ | Cluster Spearman ↑ | Spatial Moran ↑ | Contrast MAE ↓ |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = ["NA" if row["metrics"].get(k) is None else f"{row['metrics'][k]:.5f}" for k in fields]
        lines.append("| " + " | ".join([LABELS[row["task"]], row["variant"], *values]) + " |")
    lines += ["", "All metrics use the frozen benchmark test cells, catalogue, ten evaluation clusters, and masks. Spatial Moran uses the fixed source-selected panel; undefined predictions remain NA.",
              "Cluster Spearman aggregates cell predictions. context_probabilities.npy is a separate cluster-context readout, not substituted into the primary metrics.",
              "Single-step and Identity-query are retrained under the same search budget on cross-platform data. The identity-query control changes parameter count and is not perfectly capacity-matched.",
              "One seed does not support significance claims. These datasets have informed model development. Count likelihood handles finite observations but does not prove denoising or calibrated uncertainty.",
              "See the frozen pilot protocol and run_config.json for the command, device, precision, partitions, budgets, source hashes, and limitations."]
    if failures:
        lines += ["", f"Incomplete: {len(failures)} run(s) failed. Failed variants must not be omitted from later reporting."]
    (args.output_root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, default=ROOT / "outputs/unified_benchmark_20260908")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=list(LABELS), default=list(LABELS))
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["Competitive"])
    parser.add_argument("--platform-controls", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[.0003, .001])
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--cluster-weight", type=float, default=.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("A visible CUDA GPU is required; CPU fallback is forbidden")
    if min(args.epochs, args.patience, args.microbatch, args.batch_size) < 1:
        raise ValueError("Epochs, patience, and batch sizes must be positive")
    args.output_root = args.output_root.resolve()
    args.prepared_root = args.prepared_root.resolve()
    if args.output_root == args.prepared_root or args.prepared_root in args.output_root.parents:
        raise ValueError("Never write pilot files inside the frozen benchmark")
    torch.set_num_threads(4)
    threadpool_limits(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.smoke:
        args.epochs, args.patience, args.learning_rates = 2, 2, [.001]
    queue = [(task, variant) for task in args.tasks for variant in args.variants]
    if args.platform_controls and "cross_platform" in args.tasks:
        queue += [("cross_platform", v) for v in ["Single-step", "Identity-query"] if ("cross_platform", v) not in queue]
    config = {"arguments": safe_json(vars(args)), "queue": queue, "protocol_sha256": file_hash(PROTOCOL),
              "command": [sys.executable, "-m", "short2long.competitive_train", *sys.argv[1:]],
              "torch": torch.__version__, "python": sys.version, "device": args.device,
              "gpu": torch.cuda.get_device_name(args.device), "precision": "float32", "determinism": "warn_only",
              "environment": {k: os.environ.get(k) for k in ["CUDA_VISIBLE_DEVICES", "PYTHONUTF8", "PYTHONIOENCODING", "CUBLAS_WORKSPACE_CONFIG"]},
              "source_hashes": {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")},
              "prepared_manifest_hashes": {task: file_hash(args.prepared_root / "prepared" / task / "manifest.json") for task in args.tasks}}
    path = args.output_root / "run_config.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == safe_json(config), "Configuration/source changed; use a new root"
    else:
        save_json(path, config)
    snapshot = args.output_root / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for source in [*Path(__file__).parent.glob("*.py"), PROTOCOL]:
        destination = snapshot / source.name
        if destination.exists():
            assert file_hash(destination) == file_hash(source)
        else:
            shutil.copy2(source, destination)
    for task in args.tasks:
        prepared = args.prepared_root / "prepared" / task
        manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
        for name, digest in manifest["prepared_hashes"].items():
            assert file_hash(prepared / name) == digest, f"Frozen prepared artifact changed: {task}/{name}"
    save_json(args.output_root / "prepared_verification.json", {"all_prepared_hashes_verified": True, "raw_source_files_rehashed": False,
              "tasks": args.tasks, "note": "Prepared matrices checked against frozen manifests; original complete audit supplies upstream provenance."})
    rows, failures = [], []
    for position, (task, variant) in enumerate(queue, 1):
        save_json(args.output_root / "status.json", {"state": "running", "position": position, "total": len(queue),
                  "task": task, "variant": variant, "pid": os.getpid(), "updated": time.time()})
        try:
            rows.append(run_one(task, variant, args))
        except Exception:
            failure = {"task": task, "variant": variant, "traceback": traceback.format_exc()}
            failures.append(failure)
            save_json(args.output_root / "failures" / f"{task}_{variant}.json", failure)
            print(json.dumps(failure), flush=True)
        finally:
            report(args, rows, failures)
            gc.collect()
            torch.cuda.empty_cache()
    save_json(args.output_root / "status.json", {"state": "complete" if not failures else "complete_with_failures",
              "completed": len(rows), "total": len(queue), "failures": failures, "updated": time.time()})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
