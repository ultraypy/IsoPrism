from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

from .data import load_dataset, normalize_log_cpm
from .grouping import repeated_group_means
from .metrics import (
    evaluate_cluster_proportion_spearman,
    evaluate_matrix_only_predictions,
    evaluate_cluster_usage_and_switches,
    evaluate_isoform_cell_spearman,
    evaluate_pseudocell_isoform_spearman,
    evaluate_spatial_proportion_moran,
)
from .model import (
    HierarchicalIsoBudgetAttentionNet,
    IsoBudgetAttentionNet,
    IsoformAllocationNet,
)
from .train import (
    fit_capture_factors,
    parent_gene_abundance,
    predict_in_batches,
    set_seed,
    split_indices,
)


@dataclass(frozen=True)
class BenchmarkTask:
    data: str
    holdout_column: str
    holdout_value: str
    mlp_checkpoint: str
    isobudget_checkpoint: str
    hierarchical_checkpoint: str


TASKS = {
    "cross_donor": BenchmarkTask(
        data="data/processed/crc_paired",
        holdout_column="donor",
        holdout_value="PS018",
        mlp_checkpoint="outputs/crc_cross_donor_mlp_strict/model.pt",
        isobudget_checkpoint="outputs/crc_cross_donor_isobudget/model.pt",
        hierarchical_checkpoint="outputs/crc_cross_donor_hier_isobudget/model.pt",
    ),
    "cross_dataset": BenchmarkTask(
        data="data/processed/harmonized_ont_pacbio",
        holdout_column="domain",
        holdout_value="FLAMES_ONT",
        mlp_checkpoint="outputs/cross_dataset_crc_to_flames/model.pt",
        isobudget_checkpoint="outputs/cross_dataset_crc_to_flames_isobudget/model.pt",
        hierarchical_checkpoint="outputs/cross_dataset_crc_to_flames_hier_isobudget/model.pt",
    ),
    "cross_disease": BenchmarkTask(
        data="data/processed/crc_paired",
        holdout_column="condition",
        holdout_value="tumor",
        mlp_checkpoint="outputs/crc_cross_disease_mlp_strict/model.pt",
        isobudget_checkpoint="outputs/crc_cross_disease_isobudget/model.pt",
        hierarchical_checkpoint="outputs/crc_cross_disease_hier_isobudget/model.pt",
    ),
    "cross_platform": BenchmarkTask(
        data="data/processed/harmonized_ont_pacbio",
        holdout_column="domain",
        holdout_value="CRC_PacBio",
        mlp_checkpoint="outputs/cross_platform_ont_to_pacbio/model.pt",
        isobudget_checkpoint="outputs/cross_platform_ont_to_pacbio_isobudget/model.pt",
        hierarchical_checkpoint="outputs/cross_platform_ont_to_pacbio_hier_isobudget/model.pt",
    ),
    "spatial": BenchmarkTask(
        data="data/processed/post_mi_spatial",
        holdout_column="section",
        holdout_value="D",
        mlp_checkpoint="outputs/post_mi_spatial_mlp_strict/model.pt",
        isobudget_checkpoint="outputs/post_mi_spatial_isobudget/model.pt",
        hierarchical_checkpoint="outputs/post_mi_spatial_hier_isobudget/model.pt",
    ),
}

METHODS = (
    "Train-Mean",
    "PCA-kNN",
    "PCA-Ridge",
    "MLP",
    "IsoBudget-Attn",
    "Denoise-IsoBudget",
)


def _ordered_indices(catalogue: np.ndarray, selected: list[str], label: str) -> np.ndarray:
    lookup = {str(value): i for i, value in enumerate(catalogue)}
    missing = [value for value in selected if str(value) not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} {label} entries are absent from the processed matrix")
    return np.asarray([lookup[str(value)] for value in selected], dtype=np.int64)


def _training_prior(
    train_counts: sparse.csr_matrix,
    group_index: np.ndarray,
    smoothing: float = 0.5,
) -> np.ndarray:
    bulk = np.asarray(train_counts.sum(axis=0)).ravel().astype(np.float64)
    prior = np.empty_like(bulk, dtype=np.float64)
    for group in range(int(group_index.max()) + 1):
        ix = np.flatnonzero(group_index == group)
        denominator = float(bulk[ix].sum()) + smoothing * len(ix)
        prior[ix] = (bulk[ix] + smoothing) / max(denominator, 1e-12)
    return prior.astype(np.float32)


def _counts_to_probabilities(
    counts: sparse.csr_matrix | np.ndarray,
    prior: np.ndarray,
    group_index: np.ndarray,
    prior_strength: float,
) -> np.ndarray:
    values = counts.toarray() if sparse.issparse(counts) else np.asarray(counts)
    values = values.astype(np.float32, copy=False)
    probabilities = np.empty_like(values, dtype=np.float32)
    for group in range(int(group_index.max()) + 1):
        ix = np.flatnonzero(group_index == group)
        numerator = values[:, ix] + prior_strength * prior[ix][None, :]
        probabilities[:, ix] = numerator / np.maximum(
            numerator.sum(axis=1, keepdims=True), 1e-12
        )
    return probabilities


def train_mean_probabilities(
    n_evaluation_cells: int,
    train_counts: sparse.csr_matrix,
    group_index: np.ndarray,
) -> tuple[np.ndarray, dict]:
    prior = _training_prior(train_counts, group_index)
    return np.repeat(prior[None, :], n_evaluation_cells, axis=0), {
        "smoothing": 0.5,
    }


def fit_pca(
    train_expression: np.ndarray,
    evaluation_expression: np.ndarray,
    n_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    components = min(
        n_components,
        train_expression.shape[0] - 1,
        train_expression.shape[1],
    )
    model = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    train_embedding = model.fit_transform(train_expression).astype(np.float32)
    evaluation_embedding = model.transform(evaluation_expression).astype(np.float32)
    return train_embedding, evaluation_embedding, model


def short_read_evaluation_groups(
    evaluation_embedding: np.ndarray,
    seed: int,
    requested_clusters: int = 10,
    target_pseudocell_size: int = 50,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create clusters and pseudocells using held-out short reads only."""
    n_clusters = min(
        requested_clusters,
        max(2, len(evaluation_embedding) // 100),
        len(evaluation_embedding),
    )
    cluster_model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    cluster_labels = cluster_model.fit_predict(evaluation_embedding).astype(np.int64)
    pseudocell_labels = np.full(len(evaluation_embedding), -1, dtype=np.int64)
    next_label = 0
    sizes: list[int] = []
    for cluster in range(n_clusters):
        members = np.flatnonzero(cluster_labels == cluster)
        if not len(members):
            continue
        # Sorting inside a short-read cluster creates local, deterministic bins
        # without consulting held-out long-read counts.
        order = members[np.argsort(evaluation_embedding[members, 0])]
        n_pseudocells = max(1, int(round(len(order) / target_pseudocell_size)))
        for chunk in np.array_split(order, n_pseudocells):
            pseudocell_labels[chunk] = next_label
            sizes.append(int(len(chunk)))
            next_label += 1
    if np.any(pseudocell_labels < 0):
        raise RuntimeError("some evaluation cells were not assigned to a pseudocell")
    return cluster_labels, pseudocell_labels, {
        "construction_input": "held-out short-read expression in training-fitted PCA space",
        "n_clusters": int(n_clusters),
        "n_pseudocells": int(next_label),
        "target_pseudocell_size": int(target_pseudocell_size),
        "minimum_pseudocell_size": int(min(sizes)),
        "median_pseudocell_size": float(np.median(sizes)),
        "maximum_pseudocell_size": int(max(sizes)),
    }


def knn_probabilities(
    train_embedding: np.ndarray,
    evaluation_embedding: np.ndarray,
    train_counts: sparse.csr_matrix,
    group_index: np.ndarray,
    k: int,
    prior_strength: float,
) -> tuple[np.ndarray, dict]:
    neighbors = min(k, len(train_embedding))
    index = NearestNeighbors(n_neighbors=neighbors, metric="euclidean", n_jobs=-1)
    index.fit(train_embedding)
    distances, neighbor_indices = index.kneighbors(evaluation_embedding)
    weights = 1.0 / np.maximum(distances, 1e-3)
    weights = weights / weights.sum(axis=1, keepdims=True) * neighbors
    rows = np.repeat(np.arange(len(evaluation_embedding)), neighbors)
    weight_matrix = sparse.csr_matrix(
        (weights.ravel(), (rows, neighbor_indices.ravel())),
        shape=(len(evaluation_embedding), len(train_embedding)),
    )
    pooled_counts = weight_matrix @ train_counts
    prior = _training_prior(train_counts, group_index)
    probabilities = _counts_to_probabilities(
        pooled_counts, prior, group_index, prior_strength
    )
    return probabilities, {
        "k": int(neighbors),
        "distance_weighting": "inverse_euclidean",
        "effective_neighbor_weight_sum": int(neighbors),
        "prior_strength": float(prior_strength),
    }


def ridge_probabilities(
    train_embedding: np.ndarray,
    evaluation_embedding: np.ndarray,
    train_counts: sparse.csr_matrix,
    group_index: np.ndarray,
    ridge_alpha: float,
    label_smoothing: float,
    min_observed_cells: int,
) -> tuple[np.ndarray, dict]:
    prior = _training_prior(train_counts, group_index)
    probabilities = np.empty(
        (len(evaluation_embedding), train_counts.shape[1]), dtype=np.float32
    )
    fitted_groups = 0
    fallback_groups = 0
    for group in range(int(group_index.max()) + 1):
        ix = np.flatnonzero(group_index == group)
        group_counts = train_counts[:, ix].toarray().astype(np.float32)
        totals = group_counts.sum(axis=1)
        valid = totals > 0
        if valid.sum() < min_observed_cells:
            probabilities[:, ix] = prior[ix][None, :]
            fallback_groups += 1
            continue
        smoothed = group_counts[valid] + label_smoothing
        observed_probability = smoothed / smoothed.sum(axis=1, keepdims=True)
        log_ratio = np.log(np.clip(observed_probability, 1e-8, 1.0))
        log_ratio -= log_ratio.mean(axis=1, keepdims=True)
        model = Ridge(alpha=ridge_alpha, fit_intercept=True, solver="lsqr")
        sample_weight = np.minimum(totals[valid], 10.0)
        model.fit(train_embedding[valid], log_ratio, sample_weight=sample_weight)
        scores = np.asarray(model.predict(evaluation_embedding), dtype=np.float32)
        if scores.ndim == 1:
            scores = scores[:, None]
        scores -= scores.max(axis=1, keepdims=True)
        exponent = np.exp(np.clip(scores, -30, 0))
        probabilities[:, ix] = exponent / exponent.sum(axis=1, keepdims=True)
        fitted_groups += 1
    return probabilities, {
        "ridge_alpha": float(ridge_alpha),
        "label_smoothing": float(label_smoothing),
        "minimum_observed_training_cells": int(min_observed_cells),
        "fitted_genes": int(fitted_groups),
        "prior_fallback_genes": int(fallback_groups),
    }


def _load_deep_probabilities(
    checkpoint_path: Path,
    checkpoint: dict,
    expression: np.ndarray,
    context_expression: np.ndarray | None,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    group_index = torch.tensor(checkpoint["isoform_gene_index"], dtype=torch.long)
    model_type = checkpoint.get("model_type", "mlp")
    if model_type == "hierarchical_isobudget":
        model = HierarchicalIsoBudgetAttentionNet(
            n_genes=len(checkpoint["input_genes"]),
            n_isoforms=len(checkpoint["target_isoforms"]),
            isoform_gene_index=group_index,
            isoform_ids=checkpoint.get(
                "isoform_structures", checkpoint["target_isoforms"]
            ),
            prior_probabilities=checkpoint["prior_probabilities"],
            d_model=int(checkpoint["d_model"]),
            n_programs=int(checkpoint["n_programs"]),
            n_heads=int(checkpoint["n_heads"]),
            dropout=float(checkpoint["dropout"]),
            residual_gate_init=float(checkpoint.get("residual_gate_init", -1.5)),
        )
    elif model_type == "isobudget":
        model = IsoBudgetAttentionNet(
            n_genes=len(checkpoint["input_genes"]),
            n_isoforms=len(checkpoint["target_isoforms"]),
            isoform_gene_index=group_index,
            isoform_ids=checkpoint.get(
                "isoform_structures", checkpoint["target_isoforms"]
            ),
            d_model=int(checkpoint["d_model"]),
            n_programs=int(checkpoint["n_programs"]),
            n_heads=int(checkpoint["n_heads"]),
            dropout=float(checkpoint["dropout"]),
        )
    else:
        model = IsoformAllocationNet(
            n_genes=len(checkpoint["input_genes"]),
            n_isoforms=len(checkpoint["target_isoforms"]),
            isoform_gene_index=group_index,
            hidden=int(checkpoint["hidden"]),
            dropout=float(checkpoint["dropout"]),
        )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    if model_type == "hierarchical_isobudget":
        if context_expression is None or context_expression.shape != expression.shape:
            raise ValueError("hierarchical inference requires one cluster context per cell")
        batches = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(expression), batch_size):
                end = min(start + batch_size, len(expression))
                values = model(
                    torch.from_numpy(expression[start:end]).to(device),
                    torch.from_numpy(context_expression[start:end]).to(device),
                )
                batches.append(values.float().cpu())
        probabilities = torch.cat(batches).numpy()
    else:
        probabilities = predict_in_batches(model, expression, device, batch_size)
    return probabilities, {
        "checkpoint": str(checkpoint_path),
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def evaluate_method(
    true_counts: sparse.csr_matrix,
    probabilities: np.ndarray,
    group_index: np.ndarray,
    predicted_parent: sparse.csr_matrix,
    coordinates: np.ndarray | None,
    cluster_labels: np.ndarray,
    pseudocell_labels: np.ndarray,
) -> dict[str, float]:
    metrics = evaluate_matrix_only_predictions(
        true_counts,
        probabilities,
        group_index,
        predicted_parent,
    )
    metrics.update(
        evaluate_isoform_cell_spearman(
            true_counts,
            probabilities,
            group_index,
        )
    )
    metrics.update(
        evaluate_pseudocell_isoform_spearman(
            true_counts,
            probabilities,
            group_index,
            predicted_parent,
            pseudocell_labels,
        )
    )
    metrics.update(
        evaluate_cluster_usage_and_switches(
            true_counts,
            probabilities,
            group_index,
            predicted_parent,
            cluster_labels,
        )
    )
    # The paper-facing cluster metric is deliberately abundance-free: both
    # truth and prediction are unweighted means of cell-gene fractions.
    metrics.update(
        evaluate_cluster_proportion_spearman(
            true_counts,
            probabilities,
            group_index,
            cluster_labels,
        )
    )
    if coordinates is not None:
        metrics["spatial_proportion_moran_spearman"] = evaluate_spatial_proportion_moran(
            true_counts,
            probabilities,
            group_index,
            coordinates,
        )
    return metrics


def run_task(
    name: str,
    spec: BenchmarkTask,
    output_root: Path,
    methods: tuple[str, ...],
    device: torch.device,
    seed: int,
    pca_components: int,
    k: int,
    batch_size: int,
) -> dict:
    set_seed(seed)
    dataset = load_dataset(Path(spec.data))
    train_idx, eval_idx, _ = split_indices(
        dataset.obs, spec.holdout_column, spec.holdout_value, seed
    )
    isobudget_checkpoint = torch.load(
        spec.isobudget_checkpoint, map_location="cpu", weights_only=False
    )
    input_genes = [str(value) for value in isobudget_checkpoint["input_genes"]]
    target_isoforms = [str(value) for value in isobudget_checkpoint["target_isoforms"]]
    input_idx = _ordered_indices(dataset.genes, input_genes, "input gene")
    target_idx = _ordered_indices(dataset.isoforms, target_isoforms, "target isoform")
    group_index = np.asarray(
        isobudget_checkpoint["isoform_gene_index"], dtype=np.int64
    )
    target_genes = np.asarray(isobudget_checkpoint["target_genes"], dtype=str)

    expression = normalize_log_cpm(dataset.x[:, input_idx])
    train_expression = expression[train_idx]
    evaluation_expression = expression[eval_idx]
    train_counts = dataset.y[train_idx][:, target_idx].tocsr()
    evaluation_counts = dataset.y[eval_idx][:, target_idx].tocsr()
    train_parent = parent_gene_abundance(
        dataset.x[train_idx], dataset.genes, target_genes
    )
    evaluation_parent = parent_gene_abundance(
        dataset.x[eval_idx], dataset.genes, target_genes
    )
    capture_factors = fit_capture_factors(train_parent, train_counts, group_index)
    predicted_parent = evaluation_parent.multiply(capture_factors).tocsr()
    coordinates = None
    if {"spatial_x", "spatial_y"}.issubset(dataset.obs.columns):
        coordinates = dataset.obs.iloc[eval_idx][["spatial_x", "spatial_y"]].to_numpy()

    train_embedding, evaluation_embedding, pca = fit_pca(
        train_expression, evaluation_expression, pca_components, seed
    )
    pca_metadata = {
        "components": int(pca.n_components_),
        "training_explained_variance_ratio": float(
            pca.explained_variance_ratio_.sum()
        ),
        "fit_domain": "training_only",
    }
    cluster_labels, pseudocell_labels, grouping_metadata = short_read_evaluation_groups(
        evaluation_embedding, seed
    )
    evaluation_context = repeated_group_means(
        evaluation_expression, pseudocell_labels
    )

    task_report = {
        "task": name,
        "data": spec.data,
        "split": f"{spec.holdout_column}={spec.holdout_value}",
        "train_cells": int(len(train_idx)),
        "evaluation_cells": int(len(eval_idx)),
        "input_genes": int(len(input_idx)),
        "target_isoforms": int(len(target_idx)),
        "target_genes": int(group_index.max() + 1),
        "observed_evaluation_umis": int(evaluation_counts.sum()),
        "pca": pca_metadata,
        "evaluation_grouping": grouping_metadata,
        "methods": {},
    }
    for method in methods:
        started = time.time()
        if method == "Train-Mean":
            probabilities, method_metadata = train_mean_probabilities(
                len(eval_idx), train_counts, group_index
            )
        elif method == "PCA-kNN":
            probabilities, method_metadata = knn_probabilities(
                train_embedding,
                evaluation_embedding,
                train_counts,
                group_index,
                k=k,
                prior_strength=2.0,
            )
        elif method == "PCA-Ridge":
            probabilities, method_metadata = ridge_probabilities(
                train_embedding,
                evaluation_embedding,
                train_counts,
                group_index,
                ridge_alpha=10.0,
                label_smoothing=0.5,
                min_observed_cells=10,
            )
        elif method in {"MLP", "IsoBudget-Attn", "Denoise-IsoBudget"}:
            path = Path(
                spec.mlp_checkpoint
                if method == "MLP"
                else (
                    spec.isobudget_checkpoint
                    if method == "IsoBudget-Attn"
                    else spec.hierarchical_checkpoint
                )
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint["input_genes"] != input_genes:
                raise ValueError(f"{method} input catalogue differs from the canonical run")
            if checkpoint["target_isoforms"] != target_isoforms:
                raise ValueError(f"{method} target catalogue differs from the canonical run")
            probabilities, method_metadata = _load_deep_probabilities(
                path,
                checkpoint,
                evaluation_expression,
                evaluation_context if method == "Denoise-IsoBudget" else None,
                device,
                batch_size,
            )
        else:
            raise ValueError(f"unknown benchmark method: {method}")

        metrics = evaluate_method(
            evaluation_counts,
            probabilities,
            group_index,
            predicted_parent,
            coordinates,
            cluster_labels,
            pseudocell_labels,
        )
        task_report["methods"][method] = {
            "elapsed_seconds": time.time() - started,
            "metadata": method_metadata,
            "metrics": metrics,
        }
        print(
            json.dumps(
                {
                    "task": name,
                    "method": method,
                    "macro_cell_gene_jsd": metrics["macro_cell_gene_jsd"],
                    "cluster_allocation_spearman": metrics[
                        "cluster_allocation_spearman_median"
                    ],
                    "dominant_isoform_accuracy": metrics[
                        "confident_dominant_isoform_gene_macro_accuracy"
                    ],
                    "spatial_proportion_moran": metrics.get(
                        "spatial_proportion_moran_spearman"
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        del probabilities
        if device.type == "cuda":
            torch.cuda.empty_cache()

    task_output = output_root / name
    task_output.mkdir(parents=True, exist_ok=True)
    (task_output / "report.json").write_text(
        json.dumps(task_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return task_report


def write_summary(reports: list[dict], report_directory: Path) -> None:
    rows = []
    for report in reports:
        for method, result in report["methods"].items():
            row = {
                "task": report["task"],
                "split": report["split"],
                "method": method,
                "train_cells": report["train_cells"],
                "evaluation_cells": report["evaluation_cells"],
                "target_isoforms": report["target_isoforms"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
            row.update(result["metrics"])
            rows.append(row)
    frame = pd.DataFrame(rows)
    report_directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(report_directory / "matrix_only_benchmark.csv", index=False)
    (report_directory / "matrix_only_benchmark.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    labels = {
        "cross_donor": "Cross-donor",
        "cross_dataset": "Cross-dataset",
        "cross_disease": "Normal-to-tumor transfer",
        "cross_platform": "Cross-platform",
        "spatial": "Spatial transcriptomics",
    }
    lines = [
        "# Matrix-only benchmark of isoform proportion prediction",
        "",
        "Models predict within-parent-gene isoform proportions. Inference uses only short-read gene-by-cell or gene-by-spot matrices. Long-read isoform matrices define observed proportions and reliability masks. Absolute abundance, detection rates, and pseudobulk abundance correlations are not evaluated.",
        "",
        "| Task | Method | Cell-gene JSD↓ | Dominant isoform accuracy↑ | Cluster allocation Spearman↑ | Spatial proportion Moran↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        moran = row.get("spatial_proportion_moran_spearman")
        moran_text = "—" if moran is None or not np.isfinite(moran) else f"{moran:.3f}"
        lines.append(
            f"| {labels[row['task']]} | {row['method']} | "
            f"{row['macro_cell_gene_jsd']:.3f} | "
            f"{row['confident_dominant_isoform_gene_macro_accuracy']:.3f} | "
            f"{row['cluster_allocation_spearman_median']:.3f} | {moran_text} |"
        )
    lines.extend(
        [
            "",
            "## Metric definitions",
            "",
            "- Cell-gene JSD: for parent-gene LR count ≥2 UMI, normalize observed counts and predictions to within-gene proportions and average equally over eligible cell-gene pairs.",
            "- Dominant isoform accuracy: compute gene-specific accuracy for observed dominant proportion ≥0.5 and margin ≥0.1, then average equally across genes.",
            "- Cluster allocation Spearman: average observed and predicted proportions among eligible cells within clusters, concatenate multi-isoform genes, calculate Spearman correlation, and report the cluster median.",
            "- Spatial proportion Moran: compute observed and predicted proportion Moran I on spots with eligible parent-gene coverage, then correlate across isoforms.",
            "",
            "Feature selection, isoform filtering, PCA fitting, and hyperparameter selection use only the training domain.",
        ]
    )
    (report_directory / "matrix_only_benchmark.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    state_lines = [
        "# Proportion-based evaluation of cell states",
        "",
        "All metrics compare within-gene isoform proportions without multiplying by parent-gene expression. Clusters are defined by short-read expression; long reads provide proportion truth.",
        "",
        "| Task | Method | Cell-gene JSD↓ | Isoform-wise cell Spearman↑ | Cluster allocation Spearman↑ | Spatial proportion Moran↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        moran = row.get("spatial_proportion_moran_spearman")
        moran_text = "—" if moran is None or not np.isfinite(moran) else f"{moran:.3f}"
        format_metric = lambda value: (
            "NA" if value is None or not np.isfinite(value) else f"{value:.3f}"
        )
        state_lines.append(
            f"| {labels[row['task']]} | {row['method']} | {row['macro_cell_gene_jsd']:.3f} | "
            f"{format_metric(row['isoform_cell_spearman_median'])} | "
            f"{format_metric(row['cluster_allocation_spearman_median'])} | "
            f"{moran_text} |"
        )
    state_lines.extend(
        [
            "",
            "## Prespecified evaluation criteria",
            "",
            "- Isoform-wise cell correlation: parent-gene LR count ≥2 UMI, ≥30 evaluable cells, isoform detected in ≥5 cells, and Jeffreys smoothing of 0.5.",
            "- Cell-gene JSD: parent-gene LR count ≥2 UMI, with equal weighting of eligible cell-gene pairs.",
            "- Cluster Spearman: equally average proportions of eligible cells within clusters without parent-gene abundance weighting.",
            "- Spatial Moran: compute spatial autocorrelation directly on isoform proportions in coverage-eligible spots.",
        ]
    )
    (report_directory / "cell_state_benchmark.md").write_text(
        "\n".join(state_lines) + "\n", encoding="utf-8"
    )

    core_columns = [
        "task",
        "split",
        "method",
        "train_cells",
        "evaluation_cells",
        "target_isoforms",
        "confident_dominant_isoform_gene_macro_accuracy",
        "macro_cell_gene_jsd",
        "cluster_allocation_spearman_median",
        "cluster_allocation_spearman_mean",
        "cluster_allocation_spearman_min",
        "cluster_allocation_spearman_max",
        "n_clusters_allocation_spearman",
        "median_isoforms_per_cluster_spearman",
        "spatial_proportion_moran_spearman",
    ]
    available_core_columns = [column for column in core_columns if column in frame.columns]
    frame[available_core_columns].to_csv(
        report_directory / "paper_core_metrics.csv", index=False
    )
    frame[available_core_columns].to_csv(
        report_directory / "proportion_only_benchmark.csv", index=False
    )
    core_lines = [
        "# Primary metrics for within-gene isoform proportions",
        "",
        "All metrics compare within-gene proportions, not absolute isoform abundance, detection rates, or pseudobulk abundance correlations. Short reads define clusters; long reads provide proportion truth.",
        "",
        "| Task | Method | Dominant isoform accuracy↑ | Cell-gene 1−JSD↑ | Cluster allocation Spearman↑ | Spatial proportion Moran↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        format_metric = lambda value: (
            "NA" if value is None or not np.isfinite(value) else f"{value:.3f}"
        )
        core_lines.append(
            f"| {labels[row['task']]} | {row['method']} | "
            f"{format_metric(row['confident_dominant_isoform_gene_macro_accuracy'])} | "
            f"{format_metric(1.0 - row['macro_cell_gene_jsd'])} | "
            f"{format_metric(row['cluster_allocation_spearman_median'])} | "
            f"{format_metric(row.get('spatial_proportion_moran_spearman'))} |"
        )
    core_lines.extend(
        [
            "",
            "NA denotes an undefined or unavailable metric. Spatial proportion Moran requires spatial coordinates.",
        ]
    )
    (report_directory / "paper_core_metrics.md").write_text(
        "\n".join(core_lines) + "\n", encoding="utf-8"
    )

    cluster_columns = [
        "task",
        "split",
        "method",
        "evaluation_cells",
        "target_isoforms",
        "cluster_allocation_spearman_median",
        "cluster_allocation_spearman_mean",
        "cluster_allocation_spearman_min",
        "cluster_allocation_spearman_max",
        "n_clusters_allocation_spearman",
        "median_isoforms_per_cluster_spearman",
    ]
    frame[cluster_columns].to_csv(
        report_directory / "cluster_allocation_spearman.csv", index=False
    )
    point_rows = []
    for row in rows:
        values = row["cluster_allocation_spearman_values"]
        cluster_ids = row["cluster_allocation_spearman_cluster_ids"]
        for cluster_id, value in zip(cluster_ids, values):
            point_rows.append(
                {
                    "task": row["task"],
                    "method": row["method"],
                    "cluster_id": int(cluster_id),
                    "spearman": float(value),
                }
            )
    pd.DataFrame(point_rows).to_csv(
        report_directory / "cluster_allocation_spearman_points.csv", index=False
    )
    cluster_lines = [
        "# Cluster-level isoform allocation Spearman",
        "",
        "Each test domain is partitioned into ten K-means clusters in source-fitted short-read PCA space. For parent-gene LR count ≥2 UMI, compute observed and predicted within-gene proportions and average equally within clusters. Calculate Spearman correlation over concatenated multi-isoform genes and report the cluster median without expression weighting.",
        "",
        "| Task | Method | Median Spearman ↑ | Mean Spearman ↑ | Cluster range | Eligible clusters | Median isoforms per cluster |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cluster_lines.append(
            f"| {labels[row['task']]} | {row['method']} | "
            f"{row['cluster_allocation_spearman_median']:.3f} | "
            f"{row['cluster_allocation_spearman_mean']:.3f} | "
            f"{row['cluster_allocation_spearman_min']:.3f}–{row['cluster_allocation_spearman_max']:.3f} | "
            f"{row['n_clusters_allocation_spearman']} | "
            f"{row['median_isoforms_per_cluster_spearman']:.0f} |"
        )
    (report_directory / "cluster_allocation_spearman.md").write_text(
        "\n".join(cluster_lines) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/matrix_benchmarks"))
    parser.add_argument("--report-directory", type=Path, default=Path("reports"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--neighbors", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU")
    device = torch.device(args.device)
    reports = []
    for task in args.tasks:
        reports.append(
            run_task(
                task,
                TASKS[task],
                args.output_root,
                tuple(args.methods),
                device,
                args.seed,
                args.pca_components,
                args.neighbors,
                args.batch_size,
            )
        )
    write_summary(reports, args.report_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
