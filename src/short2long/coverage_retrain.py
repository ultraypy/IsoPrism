"""Retrain unchanged IsoBudget v2 on the separately versioned coverage repair.

Old checkpoints/results/code are untouched. Fixed old evaluation clusters are
used only by metrics, not by the model. Large prediction arrays are disk-backed.
"""
from __future__ import annotations

import argparse
import copy
import gc
import math
from pathlib import Path

import numpy as np
import torch
from scipy import sparse, special

from .conservative_model import ConservativeIsoBudget, choose_alpha
from .conservative_train import ALPHAS, fit, make_context, predict, source_split, validation_jsd
from .data import choose_input_genes, load_dataset, normalize_log_cpm
from .metrics import _safe_spearman
from .published_benchmark import batches, file_hash, write_json
from .train import filter_targets, retain_multi_isoform_genes, split_indices


SPECS = {
    "cross_donor": ("data/processed/crc_paired_expanded_source_v2", "donor", "PS018"),
    "cross_platform": ("data/processed/harmonized_ont_pacbio_author_v3", "domain", "CRC_PacBio"),
}


def evaluate_sparse(y, probabilities, groups, labels):
    """Same proportion estimands as metrics.py, with one gene resident at a time."""
    y = sparse.csc_matrix(y, dtype=np.float32)
    groups, labels = np.asarray(groups), np.asarray(labels)
    if probabilities.shape != y.shape:
        raise ValueError("prediction/truth shape mismatch")
    jsds, maes, accuracies, contrasts = [], [], [], []
    n_valid, n_confident, n_correct, comparisons = 0, 0, 0, 0
    observed_clusters, predicted_clusters = {}, {}
    total, nll = 0., 0.
    for gene in np.unique(groups):
        ix = np.flatnonzero(groups == gene)
        counts = y[:, ix].toarray()
        p = np.asarray(probabilities[:, ix], dtype=np.float32)
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-4):
            raise ValueError(f"invalid gene probabilities: {gene}")
        total += float(counts.sum(dtype=np.float64))
        nll += float(-(counts * np.log(np.clip(p, 1e-8, 1))).sum(dtype=np.float64))
        if len(ix) < 2:
            continue
        totals = counts.sum(1)
        valid = totals >= 2
        n_valid += int(valid.sum())
        if valid.any():
            truth = counts[valid] / totals[valid, None]
            prediction = p[valid]
            midpoint = (truth + prediction) * .5
            terms = special.xlogy(truth, truth / np.clip(midpoint, 1e-12, None)) / math.log(2)
            terms += special.xlogy(prediction, prediction / np.clip(midpoint, 1e-12, None)) / math.log(2)
            jsds.append(.5 * terms.sum(1))
            maes.append(np.abs(truth - prediction).mean(1))
            order = np.argsort(truth, axis=1)
            dominant = order[:, -1]
            top = np.take_along_axis(truth, dominant[:, None], axis=1).ravel()
            second = np.take_along_axis(truth, order[:, -2, None], axis=1).ravel()
            confident = (top >= .5) & ((top - second) >= .1)
            if confident.any():
                correct = prediction.argmax(1)[confident] == dominant[confident]
                n_confident += int(confident.sum())
                n_correct += int(correct.sum())
                accuracies.append(float(correct.mean()))
        ob, pr = [], []
        for cluster in np.unique(labels):
            keep = valid & (labels == cluster)
            if not keep.any():
                continue
            o = (counts[keep] / totals[keep, None]).mean(0)
            q = p[keep].mean(0)
            observed_clusters.setdefault(int(cluster), []).append(o)
            predicted_clusters.setdefault(int(cluster), []).append(q)
            if keep.sum() >= 5:
                ob.append(o); pr.append(q)
        errors = [np.abs((pr[a]-pr[b])-(ob[a]-ob[b])).mean()
                  for a in range(len(ob)) for b in range(a+1, len(ob))]
        if errors:
            contrasts.append(np.mean(errors)); comparisons += len(errors)
    rhos, cluster_ids, sizes = [], [], []
    for label in sorted(observed_clusters):
        observed = np.concatenate(observed_clusters[label])
        rho = _safe_spearman(observed, np.concatenate(predicted_clusters[label]))
        if np.isfinite(rho):
            rhos.append(rho); cluster_ids.append(label); sizes.append(len(observed))
    average = lambda values: float(np.mean(values)) if len(values) else None
    return {"allocation_nll": nll/max(total, 1), "observed_umis": int(total),
        "macro_cell_gene_jsd": average(np.concatenate(jsds).astype(np.float64)) if jsds else None,
        "cell_gene_proportion_mae": average(np.concatenate(maes).astype(np.float64)) if maes else None,
        "n_evaluable_cell_genes": n_valid, "n_cells": y.shape[0], "n_isoforms": y.shape[1],
        "confident_dominant_isoform_accuracy": n_correct/n_confident if n_confident else None,
        "confident_dominant_isoform_gene_macro_accuracy": average(accuracies),
        "confident_dominant_isoform_coverage": n_confident/n_valid if n_valid else None,
        "n_evaluable_dominant_pairs": n_valid, "n_confident_dominant_pairs": n_confident,
        "n_genes_confident_dominant": len(accuracies),
        "cluster_allocation_spearman_median": float(np.median(rhos)) if rhos else None,
        "cluster_allocation_spearman_mean": average(rhos),
        "cluster_allocation_spearman_values": rhos, "cluster_allocation_spearman_cluster_ids": cluster_ids,
        "n_clusters_allocation_spearman": len(rhos),
        "median_isoforms_per_cluster_spearman": float(np.median(sizes)) if sizes else None,
        "spatial_proportion_moran_spearman": None,
        "cluster_contrast_mae": average(contrasts), "contrast_evaluable_genes": len(contrasts),
        "contrast_cluster_gene_pairs": comparisons, "contrast_min_cells_per_cluster_gene": 5}


def load_repaired(task, seed, smoke_cells=None):
    path, column, holdout = SPECS[task]
    dataset = load_dataset(Path(path))
    source, test, _ = split_indices(dataset.obs, column, holdout, seed)
    # Full source only, same previously declared expanded catalogue, no top-K.
    yi = retain_multi_isoform_genes(filter_targets(dataset.y, source, 3, 2), dataset.isoform_genes)
    gi = choose_input_genes(dataset.x[source], 2000)
    old = Path("outputs/conservative_isobudget_v2") / task / "seed_2026/predicted_proportions.npz"
    with np.load(old) as saved:
        ids, fixed = saved["cell_ids"].astype(str), saved["cluster_labels"].copy()
    index = {c: i for i, c in enumerate(ids)}
    if set(dataset.obs.index[test]) != set(ids):
        raise ValueError("test cell identities changed")
    if smoke_cells:
        rng = np.random.default_rng(seed)
        partition = "donor" if task == "cross_donor" else "dataset"
        source = np.sort(np.concatenate([rng.choice(ix, min(smoke_cells, len(ix)), replace=False)
            for level in sorted(dataset.obs.iloc[source][partition].unique())
            for ix in [source[dataset.obs.iloc[source][partition].to_numpy() == level]]]))
        test = np.sort(rng.choice(test, min(64, len(test)), replace=False))
    groups = np.unique(dataset.isoform_genes[yi], return_inverse=True)[1].astype(np.int64)
    labels = np.asarray([fixed[index[c]] for c in dataset.obs.index[test]])
    metadata = {"task": task, "split": f"{column}={holdout}", "data": path,
        "train_cells": len(source), "evaluation_cells": len(test), "target_isoforms": len(yi),
        "target_genes": int(groups.max())+1, "input_genes": len(gi),
        "input_genes_list": dataset.genes[gi].tolist(), "target_isoforms_list": dataset.isoforms[yi].tolist(),
        "target_parent_gene_ids": dataset.isoform_genes[yi].tolist(),
        "isoform_gene_index": groups.tolist(), "isoform_structures": dataset.isoform_structures[yi].tolist(),
        "train_cell_ids": dataset.obs.index[source].tolist(), "evaluation_cell_ids": dataset.obs.index[test].tolist(),
        "evaluation_grouping": {"source": str(old), "labels_sha256": file_hash(old), "n_clusters": len(np.unique(labels)),
            "note": "frozen old SR-only evaluation clusters; never supplied as model context"},
        "source_catalogue_rule": ">=3 counts and >=2 detected source cells per isoform; >=2 supported isoforms/gene; no global cap",
        "internal_selection_not_nested": "input/target selection uses full source, but fitting, validation and calibration labels are disjoint",
        "data_version": dataset.metadata,
        "data_hashes": {name: file_hash(Path(path)/name) for name in ["x_gene_counts.npz", "y_isoform_counts.npz", "genes.csv", "isoforms.csv", "obs.csv", "dataset.json"]}}
    data = dict(x=normalize_log_cpm(dataset.x[source][:,gi]), y=dataset.y[source][:,yi].toarray().astype(np.float32, copy=False),
        test_x=normalize_log_cpm(dataset.x[test][:,gi]), test_y=dataset.y[test][:,yi].tocsr(),
        groups=groups, structures=metadata["isoform_structures"], clusters=labels,
        obs=dataset.obs.iloc[source].copy(), metadata=metadata)
    return data


def write_predictions(model, x, context, folder, batch_size, device):
    shape = (len(x), len(model.prior))
    outputs = {name: np.lib.format.open_memmap(folder/f"{name}.npy", mode="w+", dtype=np.float32, shape=shape)
               for name in ["raw_probabilities", "probabilities", "cluster_probabilities"]}
    model.eval()
    with torch.no_grad():
        for ix in batches(len(x), batch_size):
            raw, cluster = model(torch.from_numpy(x[ix]).to(device), torch.from_numpy(context[ix]).to(device),
                                 calibrated=False, return_components=True)
            alpha, prior = model.calibration_alpha, model.prior
            outputs["raw_probabilities"][ix] = raw.cpu().numpy()
            outputs["probabilities"][ix] = ((1-alpha)*prior + alpha*raw).cpu().numpy()
            outputs["cluster_probabilities"][ix] = ((1-alpha)*prior + alpha*cluster).cpu().numpy()
    for array in outputs.values():
        array.flush()
    return outputs


def run(task, args):
    folder = args.output_root/task/f"seed_{args.seed}"
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"refusing to overwrite a run: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "output_root": str(args.output_root), "task": task, "alphas": ALPHAS,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "code_hashes": {n: file_hash(Path(__file__).with_name(n)) for n in
          ["coverage_retrain.py", "conservative_train.py", "conservative_model.py", "model.py", "grouping.py", "metrics.py"]}}
    write_json(folder/"config.json", config)
    print(f"Loading repaired {task}", flush=True)
    data = load_repaired(task, args.seed, args.smoke_cells)
    write_json(folder/"data_manifest.json", data["metadata"])
    print({k: data["metadata"][k] for k in ["task","train_cells","evaluation_cells","target_isoforms","target_genes"]}, flush=True)
    x, y, groups, obs = data["x"], data["y"], data["groups"], data["obs"]
    train, valid, cal, strategy = source_split(obs, args.seed)
    write_json(folder/"source_split.json", {"strategy": strategy, "fit_ids": obs.index[train].tolist(),
        "validation_ids": obs.index[valid].tolist(), "calibration_ids": obs.index[cal].tolist()})
    bundle = make_context(x[train], args.seed)
    vc, cc = make_context(x[valid], args.seed)[0], make_context(x[cal], args.seed)[0]
    selected, records = None, []
    for lr in args.learning_rates:
        candidate, record = fit(x[train], y[train], groups, data["structures"], bundle, args, lr,
            folder, f"tune_{lr:g}", validation=(x[valid], y[valid], vc))
        records.append(record)
        if selected is None or record["validation_jsd"] < selected[1]["validation_jsd"]:
            selected = (copy.deepcopy(candidate).cpu(), record)
        del candidate
        torch.cuda.empty_cache()
    candidate, selection = selected
    candidate.to(args.device)
    cal_p = predict(candidate, x[cal], cc, args.batch_size, args.device, calibrated=False)
    prior = candidate.prior.detach().cpu().numpy()
    calibration = [{"alpha": a, "macro_jsd": validation_jsd((1-a)*prior+a*cal_p, y[cal], groups, args.device)} for a in ALPHAS]
    alpha = choose_alpha(calibration)
    write_json(folder/"selection.json", {"tuning": records, "selection": selection, "calibration": calibration,
        "selected_alpha": alpha, "source_split_strategy": strategy})
    del candidate, selected, cal_p, bundle, vc, cc
    gc.collect(); torch.cuda.empty_cache()
    bundle = make_context(x, args.seed)
    model, refit = fit(x, y, groups, data["structures"], bundle, args, selection["learning_rate"],
                       folder, "refit", epochs=selection["selected_epoch"])
    model.calibration_alpha.fill_(alpha)
    checkpoint = {"state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()}, "config": config,
        "metadata": data["metadata"], "selected_alpha": alpha, "input_genes": data["metadata"]["input_genes_list"],
        "groups": groups, "structures": data["structures"]}
    torch.save(checkpoint, folder/"model.pt")
    test_context, _, labels = make_context(data["test_x"], args.seed)
    prior = model.prior.detach().cpu().numpy().copy()
    result = {"task": task, "seed": args.seed, "metadata": data["metadata"], "config": config,
        "selected_alpha": alpha, "selected_learning_rate": selection["learning_rate"],
        "selected_epochs": selection["selected_epoch"], "source_split_strategy": strategy,
        "training_pseudoclusters": len(bundle[1]), "evaluation_pseudoclusters": int(labels.max())+1,
        "refit": refit, "methods": {}}
    # Fit labels are no longer needed. Keep only the independent external truth.
    del y, data["y"], bundle, checkpoint
    gc.collect()
    outputs = write_predictions(model, data["test_x"], test_context, folder, args.batch_size, args.device)
    np.savez_compressed(folder/"prediction_index.npz", context_labels=labels, cluster_labels=data["clusters"],
        cell_ids=np.asarray(data["metadata"]["evaluation_cell_ids"]), isoforms=np.asarray(data["metadata"]["target_isoforms_list"]))
    del model
    gc.collect(); torch.cuda.empty_cache()
    saved = torch.load(folder/"model.pt", map_location="cpu", weights_only=False)
    state = saved["state_dict"]
    restored = ConservativeIsoBudget(x.shape[1], groups, data["structures"], state["prior"], state["reference_expression"], d_model=args.d_model).to(args.device)
    restored.load_state_dict(state)
    max_error = 0.
    for ix in batches(len(data["test_x"]), 128):
        reproduced = predict(restored, data["test_x"][ix], test_context[ix], args.batch_size, args.device)
        expected = outputs["probabilities"][ix]
        np.testing.assert_allclose(reproduced, expected, atol=2e-6, rtol=2e-5)
        max_error = max(max_error, float(np.max(np.abs(reproduced-expected))))
    del restored, saved, state
    torch.cuda.empty_cache()
    for name, p in [("Train-Mean", np.broadcast_to(prior, outputs["probabilities"].shape)),
                    ("IsoBudget-v2-raw", outputs["raw_probabilities"]), ("IsoBudget-v2", outputs["probabilities"])]:
        print(f"Evaluating {task} {name}", flush=True)
        result["methods"][name] = evaluate_sparse(data["test_y"], p, groups, data["clusters"])
        write_json(folder/"evaluation_progress.json", result["methods"])
    result["verification"] = {"checkpoint_max_abs_error": max_error, "source_label_splits_disjoint": True,
        "external_targets_used_for_selection": False, "fixed_old_evaluation_clusters": True,
        "prediction_files": {n: file_hash(folder/n) for n in ["probabilities.npy", "raw_probabilities.npy", "cluster_probabilities.npy", "prediction_index.npz", "model.pt"]}}
    write_json(folder/"result.json", result)
    print({"task": task, "finished": True, "alpha": alpha}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=list(SPECS), default=list(SPECS))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/coverage_retrain_20260907"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[.0003, .001])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-cells", type=int)
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, no silent CPU fallback")
    torch.set_num_threads(2)
    for task in args.tasks:
        run(task, args)


if __name__ == "__main__":
    main()
