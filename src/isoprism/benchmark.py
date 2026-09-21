"""Run the eight manuscript methods on identical frozen benchmark partitions."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from short2long import competitive_train as allocation
from short2long import unified_benchmark as baselines
from short2long import competitive_structure as structure
from short2long.published_benchmark import file_hash, safe_json, verify_upstream
from short2long.unified_data import load_view, SPECS
from .report import export_tables

METHODS = ["PCA-kNN", "MLP", "Context-MLP", "TabResNet", "FT-Transformer", "BABEL", "scButterfly", "IsoPrism"]
PUBLISHED = {"TabResNet", "FT-Transformer", "BABEL", "scButterfly"}


def build_queue(tasks, methods, seeds):
    return [(task, method, seed) for task in tasks for method in methods
            for seed in (seeds[:1] if method == "PCA-kNN" else seeds)]


def verify_prepared(root, tasks):
    hashes = {}
    for task in tasks:
        folder = root / "prepared" / task
        path = folder / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("task") != task or manifest.get("smoke_cells") is not None:
            raise ValueError(f"{task}: expected a complete, task-matched frozen preparation")
        if not manifest.get("prepared_hashes"):
            raise ValueError(f"{task}: missing prepared hashes")
        source, target = manifest.get("source_ids", []), manifest.get("test_ids", [])
        if not source or not target or set(source) & set(target):
            raise ValueError(f"{task}: empty or overlapping source/target cells")
        for relative, expected in manifest["prepared_hashes"].items():
            target_path = (folder / relative).resolve()
            if folder.resolve() not in target_path.parents:
                raise ValueError("Manifest path escapes its preparation folder")
            if file_hash(target_path) != expected:
                raise ValueError(f"Changed preparation: {task}/{relative}")
        hashes[task] = file_hash(path)
    return hashes


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path.cwd(), help="Root for annotation paths")
    p.add_argument("--config", type=Path, help="Optional benchmark.json with annotation paths")
    p.add_argument("--tasks", nargs="+", choices=list(SPECS), default=list(SPECS))
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    p.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--learning-rates", nargs="+", type=float, default=[.0003, .001])
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--microbatch", type=int, default=64)
    p.add_argument("--cluster-weight", type=float, default=.5)
    p.add_argument("--device", choices=["cuda:0"], default="cuda:0",
                   help="Select the physical GPU with CUDA_VISIBLE_DEVICES")
    p.add_argument("--dry-run", action="store_true", help="Print the plan without reading data or training")
    p.add_argument("--check-only", action="store_true", help="Verify data, annotations and upstream code without training")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ("tasks", "methods", "seeds", "learning_rates"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {name} are not allowed")
    if min(args.epochs, args.patience, args.pretrain_epochs, args.d_model, args.batch_size, args.microbatch) < 1:
        raise ValueError("Training budgets and dimensions must be positive")
    if args.d_model % 4 or args.cluster_weight < 0 or min(args.learning_rates) <= 0:
        raise ValueError("Invalid dimension, loss weight or learning rate")
    planned = build_queue(args.tasks, args.methods, args.seeds)
    if args.dry_run:
        print(json.dumps({"planned_runs": len(planned), "queue": planned}, indent=2))
        return
    args.prepared_root = args.prepared_root.resolve()
    args.output_root = args.output_root.resolve()
    args.data_root = args.data_root.resolve()
    if args.output_root == args.prepared_root or args.prepared_root in args.output_root.parents or args.output_root in args.prepared_root.parents:
        raise ValueError("Use disjoint prepared and output directories")
    hashes = verify_prepared(args.prepared_root, args.tasks)
    upstream = verify_upstream() if PUBLISHED.intersection(args.methods) else {}
    annotations = dict(structure.ANNOTATIONS)
    if args.config:
        annotations.update(json.loads(args.config.read_text(encoding="utf-8"))["annotations"])
    annotation_hashes = {}
    if "IsoPrism" in args.methods:
        for task in args.tasks:
            if task in annotations:
                path = (args.data_root / annotations[task]).resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"Missing {task} annotation: {path}")
                structure.ANNOTATIONS[task] = str(path)
                annotation_hashes[task] = file_hash(path)
    if args.check_only:
        print(json.dumps({"verified_tasks": args.tasks, "prepared_hashes": hashes,
                          "annotation_hashes": annotation_hashes, "upstream": upstream}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark; inference supports CPU")
    torch.set_num_threads(4)
    threadpool_limits(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    args.smoke = False
    code_hashes = {f"{folder.name}/{path.name}": file_hash(path)
                   for folder in [Path(__file__).parent, Path(allocation.__file__).parent]
                   for path in sorted(folder.glob("*.py"))}
    config = safe_json({"arguments": {k: v for k, v in vars(args).items() if k not in {"check_only", "dry_run"}},
                       "queue": planned, "prepared_hashes": hashes, "annotation_hashes": annotation_hashes,
                       "source_hashes": code_hashes, "upstream": upstream, "torch": torch.__version__,
                       "python": sys.version, "gpu": torch.cuda.get_device_name(0)})
    run_config = args.output_root / "run_config.json"
    if run_config.exists():
        if json.loads(run_config.read_text(encoding="utf-8")) != config:
            raise ValueError("Configuration/source changed; use a new output root")
    elif args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError("Output directory contains unrecognized files")
    else:
        allocation.save_json(run_config, config)
    if "IsoPrism" in args.methods:
        for task in args.tasks:
            for stage in ("inner", "final"):
                folder = args.prepared_root / "prepared" / task / stage
                values, metadata = structure.terminal_features(task, load_view(folder), args.data_root)
                record = safe_json({"features": values, "metadata": metadata,
                                    "catalogue_sha256": file_hash(folder / "catalogue.json")})
                destination = args.output_root / "structural_features" / task / f"{stage}.json"
                if destination.exists() and json.loads(destination.read_text(encoding="utf-8")) != record:
                    raise ValueError("Structural features changed")
                allocation.save_json(destination, record)
    completed, failures = [], []
    for task, method, seed in planned:
        allocation.save_json(args.output_root / "status.json", {"state": "running", "completed": len(completed),
                             "planned": len(planned), "task": task, "method": method, "seed": seed})
        try:
            if method == "IsoPrism":
                args.seed = seed
                allocation.run_one(task, method, args)
            else:
                baselines.run_one(task, method, seed, args)
            completed.append((task, method, seed))
        except Exception:
            failures.append({"task": task, "method": method, "seed": seed, "traceback": traceback.format_exc()})
            allocation.save_json(args.output_root / "failures.json", failures)
            raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            export_tables(args.output_root)
            allocation.save_json(args.output_root / "status.json", {"state": "failed" if failures else "running",
                                 "completed": len(completed), "planned": len(planned), "failures": failures})
    allocation.save_json(args.output_root / "status.json", {"state": "complete", "completed": len(completed), "planned": len(planned)})


def train_main():
    if "--methods" in sys.argv[1:]:
        raise SystemExit("isoprism-train always trains IsoPrism; use isoprism-benchmark for comparators")
    main([*sys.argv[1:], "--methods", "IsoPrism"])


if __name__ == "__main__":
    main()
