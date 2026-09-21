"""Fixed-protocol source-selected improvement; external labels only enter evaluation."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from .conservative_model import (ConservativeIsoBudget, cell_targets, choose_alpha,
                                 cluster_contrast_metrics, macro_cross_entropy, macro_jsd)
from .grouping import build_pseudocluster_labels, group_means
from .hierarchical_train import training_prior
from .matrix_benchmark import TASKS, train_mean_probabilities
from .published_benchmark import batches, evaluate, file_hash, load_task, write_json
from .train import set_seed


ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


def source_split(obs, seed):
    """Fit / early stopping / correction calibration are label-disjoint."""
    rng = np.random.default_rng(seed)
    column = None
    for candidate in ("donor", "section", "dataset", "sample"):
        if candidate in obs and obs[candidate].notna().all() and obs[candidate].nunique() >= 2:
            column = candidate
            break
    if column:
        values = obs[column].astype(str).to_numpy()
        levels = np.asarray(sorted(np.unique(values)))
        rng.shuffle(levels)
        if len(levels) >= 3:
            n = min(max(1, int(round(len(levels) * 0.15))), (len(levels) - 1) // 2)
            calibration = np.flatnonzero(np.isin(values, levels[:n]))
            validation = np.flatnonzero(np.isin(values, levels[n:2*n]))
            fit = np.flatnonzero(~np.isin(values, levels[:2*n]))
            strategy = f"disjoint {column} groups for fit/validation/calibration"
        else:
            fit = np.flatnonzero(values != levels[0])
            heldout = rng.permutation(np.flatnonzero(values == levels[0]))
            validation, calibration = np.array_split(heldout, 2)
            strategy = f"held-out {column}; validation/calibration split cells within that group"
    else:
        shuffled = rng.permutation(len(obs))
        n = max(1, int(round(len(obs) * 0.1)))
        validation, calibration, fit = shuffled[:n], shuffled[n:2*n], shuffled[2*n:]
        strategy = "random 80/10/10 cells; no suitable biological grouping"
    if min(map(len, (fit, validation, calibration))) < 2:
        raise ValueError("Source split is too small; do not use external labels as a fallback")
    assert not (set(fit) & set(validation) or set(fit) & set(calibration) or set(validation) & set(calibration))
    return np.sort(fit), np.sort(validation), np.sort(calibration), strategy


def make_context(x, seed, target_size=50):
    labels, _ = build_pseudocluster_labels(x, seed=seed, target_size=target_size)
    means = group_means(x, labels)
    return means[labels], means, labels


def cluster_targets(y, labels, groups, prior):
    n_groups = int(groups.max()) + 1
    totals = np.zeros((len(y), n_groups), dtype=np.float32)
    np.add.at(totals.T, groups, y.T)
    valid = totals >= 2
    fractions = y / np.maximum(totals[:, groups], 1e-8)
    membership = sparse.csr_matrix((np.ones(len(labels), np.float32), (labels, np.arange(len(labels)))),
                                   shape=(int(labels.max()) + 1, len(labels)))
    coverage = np.asarray(membership @ valid.astype(np.float32))
    summed = np.asarray(membership @ (fractions * valid[:, groups]))
    # Equal-cell fractions with five equivalent prior observations per cluster-gene.
    targets = (summed + 5.0 * prior[None, :]) / (coverage[:, groups] + 5.0)
    return targets.astype(np.float32), coverage >= 3


def predict(model, x, context, batch_size, device, calibrated=True, components=False):
    model.eval()
    output, cluster_output = [], []
    with torch.no_grad():
        for ix in batches(len(x), batch_size):
            p, cp = model(torch.from_numpy(x[ix]).to(device), torch.from_numpy(context[ix]).to(device),
                          calibrated=calibrated, return_components=True)
            output.append(p.cpu().numpy())
            if components:
                cluster_output.append(cp.cpu().numpy())
    result = np.concatenate(output)
    return (result, np.concatenate(cluster_output)) if components else result


def validation_jsd(p, y, groups, device):
    total, count = 0.0, 0
    g = torch.as_tensor(groups, device=device)
    with torch.no_grad():
        for ix in batches(len(p), 256):
            value, n = macro_jsd(torch.from_numpy(p[ix]).to(device), torch.from_numpy(y[ix]).to(device), g)
            total += float(value)
            count += int(n)
    if not count:
        raise ValueError("No source validation cell-gene with parent count >= 2")
    return total / count


def fit(x, y, groups, structures, bundle, args, lr, folder, tag, validation=None, epochs=None):
    set_seed(args.seed)
    device = torch.device(args.device)
    prior = training_prior(y, groups)
    model = ConservativeIsoBudget(x.shape[1], groups, structures, prior, x.mean(0),
                                  d_model=args.d_model).to(device)
    context, cluster_x, labels = bundle
    cy, cv = cluster_targets(y, labels, groups, prior)
    cx = torch.from_numpy(cluster_x).to(device)
    ct, cm = torch.from_numpy(cy).to(device), torch.from_numpy(cv).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    history, best_score, best_state, best_epoch, stale = [], float("inf"), None, 0, 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, (epochs or args.epochs) + 1):
        model.train()
        loss_sum, batch_count = 0.0, 0
        for ix in batches(len(x), args.batch_size, rng):
            xb, cb, yb = [torch.from_numpy(a[ix]).to(device) for a in (x, context, y)]
            cix = torch.as_tensor(rng.choice(len(cx), min(8, len(cx)), replace=False), device=device)
            optimizer.zero_grad(set_to_none=True)
            p = model(xb, cb, calibrated=False)
            targets, valid = cell_targets(yb, model.isoform_gene_index, model.prior)
            cell_loss = macro_cross_entropy(p, targets, valid, model.isoform_gene_index)
            cp = model(cx[cix], cx[cix], calibrated=False)
            cluster_loss = macro_cross_entropy(cp, ct[cix], cm[cix], model.isoform_gene_index)
            anchor = (p * (p.clamp_min(1e-8).log() - model.prior.log())).sum() / (len(p) * model.n_target_genes)
            loss = cell_loss + 0.5 * cluster_loss + 0.02 * anchor
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {tag} loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            batch_count += 1
        score = None
        if validation is not None:
            vx, vy, vc = validation
            vp = predict(model, vx, vc, args.batch_size, device, calibrated=False)
            score = validation_jsd(vp, vy, groups, device)
            if score < best_score - 1e-7:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
        history.append({"epoch": epoch, "training_loss": loss_sum / batch_count,
                        "validation_macro_jsd": score, "seconds": time.perf_counter() - started})
        write_json(folder / f"progress_{tag}.json", history)
        print(json.dumps({"stage": tag, **history[-1]}), flush=True)
        if validation is not None and stale >= args.patience:
            break
    if validation is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epoch
    record = {"learning_rate": lr, "selected_epoch": best_epoch, "validation_jsd": best_score if validation else None,
              "history": history, "seconds": time.perf_counter() - started,
              "parameters": sum(p.numel() for p in model.parameters()),
              "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2}
    return model, record


def run(task, args):
    folder = args.output_root / task / f"seed_{args.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "task": task, "output_root": str(args.output_root), "alphas": ALPHAS,
              "model_hash": file_hash(Path(__file__).with_name("conservative_model.py")),
              "trainer_hash": file_hash(__file__), "base_model_hash": file_hash(Path(__file__).with_name("model.py")),
              "grouping_hash": file_hash(Path(__file__).with_name("grouping.py")),
              "metrics_hash": file_hash(Path(__file__).with_name("metrics.py")),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    if (folder / "result.json").exists():
        saved = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if saved != config:
            raise ValueError("Refusing to overwrite a completed run with a different configuration")
        print(f"Skip completed {task}", flush=True)
        return
    write_json(folder / "config.json", config)
    data = load_task(task, split_seed=args.seed, smoke_cells=args.smoke_cells)
    x, y, groups = data["x"], data["y"], data["groups"]
    obs = pd.read_csv(Path(data["metadata"]["data"]) / "obs.csv", index_col=0)
    obs = obs.loc[data["metadata"]["train_cell_ids"]]
    train, valid, cal, strategy = source_split(obs, args.seed)
    write_json(folder / "source_split.json", {"strategy": strategy, "fit_indices": train,
                "validation_indices": valid, "calibration_indices": cal,
                "fit_ids": obs.index[train].tolist(), "validation_ids": obs.index[valid].tolist(),
                "calibration_ids": obs.index[cal].tolist()})
    train_bundle = make_context(x[train], args.seed)
    val_context = make_context(x[valid], args.seed)[0]
    cal_context = make_context(x[cal], args.seed)[0]
    selected, records = None, []
    for lr in args.learning_rates:
        candidate, record = fit(x[train], y[train], groups, data["structures"], train_bundle,
                                args, lr, folder, f"tune_{lr:g}",
                                validation=(x[valid], y[valid], val_context))
        records.append(record)
        if selected is None or record["validation_jsd"] < selected[1]["validation_jsd"]:
            selected = (copy.deepcopy(candidate).cpu(), record)
        del candidate
        torch.cuda.empty_cache()
    candidate, selection = selected
    candidate = candidate.to(args.device)
    cal_p = predict(candidate, x[cal], cal_context, args.batch_size, args.device, calibrated=False)
    prior = candidate.prior.detach().cpu().numpy()
    calibration = [{"alpha": alpha, "macro_jsd": validation_jsd(
        (1-alpha)*prior + alpha*cal_p, y[cal], groups, args.device)} for alpha in ALPHAS]
    alpha = choose_alpha(calibration)
    write_json(folder / "selection.json", {"tuning": records, "selection": selection,
                "calibration": calibration, "selected_alpha": alpha, "source_split_strategy": strategy})
    del candidate, selected, cal_p
    torch.cuda.empty_cache()
    bundle = make_context(x, args.seed)
    model, refit = fit(x, y, groups, data["structures"], bundle, args,
                        selection["learning_rate"], folder, "refit", epochs=selection["selected_epoch"])
    model.calibration_alpha.fill_(alpha)
    checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                   "config": config, "metadata": data["metadata"], "selected_alpha": alpha,
                   "input_genes": data["metadata"]["input_genes_list"],
                   "groups": groups, "structures": data["structures"]}
    torch.save(checkpoint, folder / "model.pt")
    # Only now is the external expression used for prediction and LR for evaluation.
    test_context, _, labels = make_context(data["test_x"], args.seed)
    raw, raw_cluster = predict(model, data["test_x"], test_context, args.batch_size, args.device,
                                calibrated=False, components=True)
    prior = model.prior.detach().cpu().numpy()
    calibrated = (1-alpha)*prior + alpha*raw
    cluster_p = (1-alpha)*prior + alpha*raw_cluster
    mean, _ = train_mean_probabilities(len(calibrated), y, groups)
    result = {"task": task, "seed": args.seed, "metadata": data["metadata"], "config": config,
              "selected_alpha": alpha, "selected_learning_rate": selection["learning_rate"],
              "selected_epochs": selection["selected_epoch"], "source_split_strategy": strategy,
              "training_pseudoclusters": len(bundle[1]), "evaluation_pseudoclusters": int(labels.max()) + 1,
              "refit": refit, "methods": {}}
    for name, p in (("Train-Mean", mean), ("IsoBudget-v2-raw", raw), ("IsoBudget-v2", calibrated)):
        metrics = evaluate(data, p)
        metrics.update(cluster_contrast_metrics(data["test_y"], p, groups, data["clusters"]))
        result["methods"][name] = metrics
    np.savez_compressed(folder / "predicted_proportions.npz", probabilities=calibrated,
                         raw_probabilities=raw, cluster_probabilities=cluster_p,
                         context_labels=labels, cluster_labels=data["clusters"],
                         cell_ids=np.asarray(data["metadata"]["evaluation_cell_ids"]),
                         isoforms=np.asarray(data["metadata"]["target_isoforms_list"]))
    restored = ConservativeIsoBudget(x.shape[1], groups, data["structures"], prior, x.mean(0),
                                     d_model=args.d_model).to(args.device)
    restored.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=False)["state_dict"])
    reproduced = predict(restored, data["test_x"], test_context, args.batch_size, args.device)
    np.testing.assert_allclose(reproduced, calibrated, atol=2e-6, rtol=2e-5)
    result["verification"] = {"checkpoint_max_abs_error": float(np.abs(reproduced-calibrated).max()),
                               "source_label_splits_disjoint": True, "external_targets_used_for_selection": False}
    write_json(folder / "result.json", result)
    print(json.dumps({"task": task, "finished": True, "alpha": alpha}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/conservative_isobudget_v2"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.0003, 0.001])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-cells", type=int)
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA training is required; no implicit CPU fallback")
    torch.set_num_threads(2)
    for task in args.tasks:
        run(task, args)


if __name__ == "__main__":
    main()
