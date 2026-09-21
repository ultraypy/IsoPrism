from __future__ import annotations

import math

import numpy as np
from scipy import sparse, special, stats
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, matthews_corrcoef


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.ptp(a[keep]) == 0 or np.ptp(b[keep]) == 0:
        return float("nan")
    return float(stats.spearmanr(a[keep], b[keep]).statistic)


def evaluate_confident_dominant_accuracy(
    true_counts: sparse.csr_matrix | np.ndarray,
    predicted_scores: np.ndarray,
    group_index: np.ndarray,
    min_parent_umis: int = 2,
    min_dominant_fraction: float = 0.5,
    min_dominant_margin: float = 0.1,
) -> dict[str, float]:
    """Score the dominant isoform only when the observed winner is reliable.

    Coverage is the fraction of evaluable multi-isoform cell/cluster-gene pairs
    that pass the observed dominant-fraction and first-versus-second margin
    thresholds. The gene-macro score gives every parent gene equal weight.
    """
    y = true_counts.toarray() if sparse.issparse(true_counts) else np.asarray(true_counts)
    y = y.astype(np.float32, copy=False)
    predicted_scores = np.asarray(predicted_scores, dtype=np.float32)
    group_index = np.asarray(group_index, dtype=np.int64)
    if predicted_scores.shape != y.shape:
        raise ValueError(
            f"predicted score shape {predicted_scores.shape} != truth shape {y.shape}"
        )

    n_evaluable = 0
    n_confident = 0
    n_correct = 0
    gene_accuracies: list[float] = []
    for group in range(int(group_index.max()) + 1):
        ix = np.flatnonzero(group_index == group)
        if len(ix) < 2:
            continue
        observed = y[:, ix]
        observed_total = observed.sum(axis=1)
        valid = observed_total >= min_parent_umis
        n_evaluable += int(valid.sum())
        if not valid.any():
            continue

        observed = observed[valid]
        observed_probability = observed / observed.sum(axis=1, keepdims=True)
        observed_order = np.argsort(observed_probability, axis=1)
        observed_dominant = observed_order[:, -1]
        top = np.take_along_axis(
            observed_probability, observed_dominant[:, None], axis=1
        ).ravel()
        second = np.take_along_axis(
            observed_probability, observed_order[:, -2, None], axis=1
        ).ravel()
        confident = (top >= min_dominant_fraction) & (
            (top - second) >= min_dominant_margin
        )
        if not confident.any():
            continue

        predicted_dominant = np.argmax(predicted_scores[valid][:, ix], axis=1)
        correct = predicted_dominant[confident] == observed_dominant[confident]
        n_confident += int(confident.sum())
        n_correct += int(correct.sum())
        gene_accuracies.append(float(correct.mean()))

    return {
        "confident_dominant_isoform_accuracy": (
            float(n_correct / n_confident) if n_confident else float("nan")
        ),
        "confident_dominant_isoform_gene_macro_accuracy": (
            float(np.mean(gene_accuracies)) if gene_accuracies else float("nan")
        ),
        "confident_dominant_isoform_coverage": (
            float(n_confident / n_evaluable) if n_evaluable else float("nan")
        ),
        "n_evaluable_dominant_pairs": int(n_evaluable),
        "n_confident_dominant_pairs": int(n_confident),
        "n_genes_confident_dominant": int(len(gene_accuracies)),
    }


def grouped_pseudobulk_jsd(
    true_counts: np.ndarray, predicted_counts: np.ndarray, group_index: np.ndarray
) -> tuple[float, float]:
    true_bulk = true_counts.sum(axis=0).astype(np.float64)
    pred_bulk = predicted_counts.sum(axis=0).astype(np.float64)
    jsds: list[float] = []
    weights: list[float] = []
    for group in np.unique(group_index):
        ix = group_index == group
        if ix.sum() < 2:
            continue
        t = true_bulk[ix]
        p = pred_bulk[ix]
        if t.sum() <= 0:
            continue
        if p.sum() <= 0:
            jsds.append(1.0)
            weights.append(float(true_bulk[ix].sum()))
            continue
        t /= t.sum()
        p /= p.sum()
        m = 0.5 * (t + p)
        jsd = 0.5 * stats.entropy(t, m, base=2) + 0.5 * stats.entropy(p, m, base=2)
        jsds.append(float(jsd))
        weights.append(float(true_bulk[ix].sum()))
    if not jsds:
        return float("nan"), float("nan")
    return float(np.median(jsds)), float(np.average(jsds, weights=weights))


def evaluate_allocations(
    true_counts: sparse.csr_matrix | np.ndarray,
    probabilities: np.ndarray,
    group_index: np.ndarray,
    parent_abundance: sparse.csr_matrix | np.ndarray | None = None,
) -> dict[str, float]:
    y = true_counts.toarray() if sparse.issparse(true_counts) else np.asarray(true_counts)
    y = y.astype(np.float32, copy=False)
    total = float(y.sum())
    nll = float(-(y * np.log(np.clip(probabilities, 1e-8, 1))).sum() / max(total, 1.0))

    # The network predicts P(isoform | gene, cell).  For end-to-end evaluation,
    # multiply by short-read parent-gene abundance and scale only the per-cell
    # library depth to the observed long-read total.  The fallback is retained
    # for unit tests and allocation-only diagnostics.
    gene_totals = np.zeros((y.shape[0], int(group_index.max()) + 1), dtype=np.float32)
    np.add.at(gene_totals.T, group_index, y.T)
    if parent_abundance is None:
        scaled_parent = gene_totals
    else:
        scaled_parent = (
            parent_abundance.toarray()
            if sparse.issparse(parent_abundance)
            else np.asarray(parent_abundance)
        ).astype(np.float32, copy=False)
        if scaled_parent.shape != gene_totals.shape:
            raise ValueError(
                f"parent abundance shape {scaled_parent.shape} != {gene_totals.shape}"
            )
        scaled_parent = np.maximum(scaled_parent, 0)
        short_total = scaled_parent.sum(axis=1)
        long_total = y.sum(axis=1)
        scale = long_total / np.maximum(short_total, 1e-8)
        scaled_parent = scaled_parent * scale[:, None]
    pred_counts = probabilities * scaled_parent[:, group_index]
    true_bulk = y.sum(axis=0)
    pred_bulk = pred_counts.sum(axis=0)
    allocation_bulk = (probabilities * gene_totals[:, group_index]).sum(axis=0)
    group_sizes = np.bincount(group_index)
    multi_isoform = group_sizes[group_index] >= 2
    pseudo_spearman = _safe_spearman(true_bulk[multi_isoform], pred_bulk[multi_isoform])
    allocation_spearman = _safe_spearman(
        true_bulk[multi_isoform], allocation_bulk[multi_isoform]
    )
    median_jsd, weighted_jsd = grouped_pseudobulk_jsd(y, pred_counts, group_index)

    top1 = np.zeros(probabilities.shape[1], dtype=bool)
    top3 = np.zeros(probabilities.shape[1], dtype=bool)
    for group in np.unique(group_index):
        ix = np.flatnonzero(group_index == group)
        if len(ix) < 2:
            continue
        order = ix[np.argsort(pred_bulk[ix])[::-1]]
        top1[order[:1]] = True
        top3[order[: min(3, len(order))]] = True
    multi_total = float(y[:, multi_isoform].sum())
    top1_mass = float(y[:, top1].sum() / max(multi_total, 1.0))
    top3_mass = float(y[:, top3].sum() / max(multi_total, 1.0))

    valid_dominant = 0
    correct_dominant = 0
    for group in np.unique(group_index):
        ix = np.flatnonzero(group_index == group)
        if len(ix) < 2:
            continue
        valid = gene_totals[:, group] >= 2
        if not valid.any():
            continue
        correct_dominant += int(
            (np.argmax(y[valid][:, ix], axis=1) == np.argmax(probabilities[valid][:, ix], axis=1)).sum()
        )
        valid_dominant += int(valid.sum())

    # Detection is scored only where the parent gene was observed, avoiding a
    # trivial reward from the many cell/gene pairs with zero long-read coverage.
    detection_mask = (gene_totals[:, group_index] > 0) & multi_isoform[None, :]
    observed = (y > 0)[detection_mask]
    scores = pred_counts[detection_mask]
    auprc = float(average_precision_score(observed, scores)) if observed.any() else float("nan")
    prevalence = float(observed.mean())
    metrics = {
        "allocation_nll": nll,
        "allocation_perplexity": float(math.exp(min(nll, 30))),
        "pseudobulk_isoform_spearman": pseudo_spearman,
        "allocation_pseudobulk_isoform_spearman": allocation_spearman,
        "median_gene_jsd": median_jsd,
        "weighted_gene_jsd": weighted_jsd,
        "dominant_isoform_accuracy": float(correct_dominant / max(valid_dominant, 1)),
        "top1_observed_umi_mass": top1_mass,
        "top3_observed_umi_mass": top3_mass,
        "detection_auprc": auprc,
        "detection_prevalence": prevalence,
        "detection_auprc_lift": auprc - prevalence,
        "n_cells": int(y.shape[0]),
        "n_isoforms": int(y.shape[1]),
        "observed_umis": int(total),
    }
    return metrics


def evaluate_matrix_only_predictions(
    true_counts: sparse.csr_matrix | np.ndarray,
    probabilities: np.ndarray,
    group_index: np.ndarray,
    predicted_parent_abundance: sparse.csr_matrix | np.ndarray,
    min_gene_umis: int = 2,
) -> dict[str, float]:
    """Evaluate predictions without using held-out long reads as model input.

    ``predicted_parent_abundance`` must be computed exclusively from the short-read
    matrix and training-domain calibration. In contrast to ``evaluate_allocations``,
    this function never rescales a held-out cell by its observed long-read library
    size. The long-read matrix is read only to construct evaluation targets.
    """
    y = true_counts.toarray() if sparse.issparse(true_counts) else np.asarray(true_counts)
    y = y.astype(np.float32, copy=False)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    group_index = np.asarray(group_index, dtype=np.int64)
    parent = (
        predicted_parent_abundance.toarray()
        if sparse.issparse(predicted_parent_abundance)
        else np.asarray(predicted_parent_abundance)
    ).astype(np.float32, copy=False)
    n_groups = int(group_index.max()) + 1
    if probabilities.shape != y.shape:
        raise ValueError(f"probability shape {probabilities.shape} != truth shape {y.shape}")
    if parent.shape != (len(y), n_groups):
        raise ValueError(f"parent abundance shape {parent.shape} != {(len(y), n_groups)}")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")

    group_sums = np.zeros((len(y), n_groups), dtype=np.float32)
    np.add.at(group_sums.T, group_index, probabilities.T)
    if not np.allclose(group_sums, 1.0, atol=1e-4):
        raise ValueError("isoform probabilities must sum to one inside every gene")

    total = float(y.sum())
    nll = float(-(y * np.log(np.clip(probabilities, 1e-8, 1))).sum() / max(total, 1.0))
    parent = np.maximum(parent, 0)
    predicted_counts = probabilities * parent[:, group_index]
    true_bulk = y.sum(axis=0).astype(np.float64)
    predicted_bulk = predicted_counts.sum(axis=0).astype(np.float64)
    pseudobulk_spearman = _safe_spearman(true_bulk, predicted_bulk)

    true_fraction = true_bulk / max(float(true_bulk.sum()), 1.0)
    predicted_fraction = predicted_bulk / max(float(predicted_bulk.sum()), 1.0)
    true_log_cpm = np.log1p(true_fraction * 1e6)
    predicted_log_cpm = np.log1p(predicted_fraction * 1e6)
    if np.ptp(true_log_cpm) == 0 or np.ptp(predicted_log_cpm) == 0:
        pseudobulk_pearson = float("nan")
    else:
        pseudobulk_pearson = float(stats.pearsonr(true_log_cpm, predicted_log_cpm).statistic)

    median_gene_jsd, weighted_gene_jsd = grouped_pseudobulk_jsd(
        y, predicted_counts, group_index
    )
    true_gene_totals = np.zeros((len(y), n_groups), dtype=np.float32)
    np.add.at(true_gene_totals.T, group_index, y.T)

    cell_gene_jsds: list[np.ndarray] = []
    cell_gene_weights: list[np.ndarray] = []
    valid_dominant = 0
    correct_dominant = 0
    absolute_errors: list[np.ndarray] = []
    for group in range(n_groups):
        ix = np.flatnonzero(group_index == group)
        valid = true_gene_totals[:, group] >= min_gene_umis
        if len(ix) < 2 or not valid.any():
            continue
        observed = y[valid][:, ix]
        observed_probability = observed / observed.sum(axis=1, keepdims=True)
        predicted_probability = probabilities[valid][:, ix]
        midpoint = 0.5 * (observed_probability + predicted_probability)
        observed_term = special.xlogy(
            observed_probability,
            observed_probability / np.clip(midpoint, 1e-12, None),
        ) / math.log(2)
        predicted_term = special.xlogy(
            predicted_probability,
            predicted_probability / np.clip(midpoint, 1e-12, None),
        ) / math.log(2)
        cell_gene_jsds.append(0.5 * (observed_term + predicted_term).sum(axis=1))
        cell_gene_weights.append(true_gene_totals[valid, group])
        absolute_errors.append(np.abs(observed_probability - predicted_probability).mean(axis=1))
        correct_dominant += int(
            (np.argmax(observed, axis=1) == np.argmax(predicted_probability, axis=1)).sum()
        )
        valid_dominant += int(valid.sum())

    if cell_gene_jsds:
        jsd_values = np.concatenate(cell_gene_jsds).astype(np.float64)
        jsd_weights = np.concatenate(cell_gene_weights).astype(np.float64)
        macro_cell_gene_jsd = float(jsd_values.mean())
        weighted_cell_gene_jsd = float(np.average(jsd_values, weights=jsd_weights))
        cell_gene_mae = float(np.concatenate(absolute_errors).mean())
        n_evaluable = int(len(jsd_values))
    else:
        macro_cell_gene_jsd = float("nan")
        weighted_cell_gene_jsd = float("nan")
        cell_gene_mae = float("nan")
        n_evaluable = 0

    group_sizes = np.bincount(group_index, minlength=n_groups)
    multi_isoform = group_sizes[group_index] >= 2
    detection_mask = (true_gene_totals[:, group_index] > 0) & multi_isoform[None, :]
    observed_detection = (y > 0)[detection_mask]
    detection_scores = predicted_counts[detection_mask]
    if observed_detection.any():
        detection_auprc = float(average_precision_score(observed_detection, detection_scores))
        detection_prevalence = float(observed_detection.mean())
    else:
        detection_auprc = float("nan")
        detection_prevalence = float("nan")

    metrics = {
        "allocation_nll": nll,
        "allocation_perplexity": float(math.exp(min(nll, 30))),
        "weighted_cell_gene_jsd": weighted_cell_gene_jsd,
        "macro_cell_gene_jsd": macro_cell_gene_jsd,
        "cell_gene_proportion_mae": cell_gene_mae,
        "dominant_isoform_accuracy": float(correct_dominant / max(valid_dominant, 1)),
        "pseudobulk_isoform_spearman": pseudobulk_spearman,
        "pseudobulk_log_cpm_pearson": pseudobulk_pearson,
        "median_gene_jsd": median_gene_jsd,
        "weighted_gene_jsd": weighted_gene_jsd,
        "detection_auprc": detection_auprc,
        "detection_prevalence": detection_prevalence,
        "detection_auprc_lift": detection_auprc - detection_prevalence,
        "n_evaluable_cell_genes": n_evaluable,
        "n_cells": int(y.shape[0]),
        "n_isoforms": int(y.shape[1]),
        "observed_umis": int(total),
    }
    metrics.update(
        evaluate_confident_dominant_accuracy(
            y,
            probabilities,
            group_index,
            min_parent_umis=min_gene_umis,
        )
    )
    sensitivity = evaluate_confident_dominant_accuracy(
        y,
        probabilities,
        group_index,
        min_parent_umis=5,
    )
    metrics.update({f"{key}_min5": value for key, value in sensitivity.items()})
    return metrics


def evaluate_proportion_predictions(
    true_counts: sparse.csr_matrix | np.ndarray,
    probabilities: np.ndarray,
    group_index: np.ndarray,
    min_gene_umis: int = 2,
) -> dict[str, float]:
    """Evaluate only within-parent-gene isoform proportions.

    Long-read counts define reliable observed fractions and evaluation masks, but
    neither short-read parent abundance nor predicted isoform abundance enters a
    metric. Cell-gene divergences are macro-averaged so every evaluable fraction
    vector has equal weight.
    """
    y = true_counts.toarray() if sparse.issparse(true_counts) else np.asarray(true_counts)
    y = y.astype(np.float32, copy=False)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    group_index = np.asarray(group_index, dtype=np.int64)
    if probabilities.shape != y.shape:
        raise ValueError(f"probability shape {probabilities.shape} != truth shape {y.shape}")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")

    n_groups = int(group_index.max()) + 1
    probability_sums = np.zeros((len(y), n_groups), dtype=np.float32)
    np.add.at(probability_sums.T, group_index, probabilities.T)
    if not np.allclose(probability_sums, 1.0, atol=1e-4):
        raise ValueError("isoform probabilities must sum to one inside every gene")

    gene_totals = np.zeros((len(y), n_groups), dtype=np.float32)
    np.add.at(gene_totals.T, group_index, y.T)
    total = float(y.sum())
    nll = float(
        -(y * np.log(np.clip(probabilities, 1e-8, 1))).sum() / max(total, 1.0)
    )
    jsd_parts: list[np.ndarray] = []
    mae_parts: list[np.ndarray] = []
    for group in range(n_groups):
        ix = np.flatnonzero(group_index == group)
        valid = gene_totals[:, group] >= min_gene_umis
        if len(ix) < 2 or not valid.any():
            continue
        observed = y[valid][:, ix]
        observed_probability = observed / observed.sum(axis=1, keepdims=True)
        predicted_probability = probabilities[valid][:, ix]
        midpoint = 0.5 * (observed_probability + predicted_probability)
        observed_term = special.xlogy(
            observed_probability,
            observed_probability / np.clip(midpoint, 1e-12, None),
        ) / math.log(2)
        predicted_term = special.xlogy(
            predicted_probability,
            predicted_probability / np.clip(midpoint, 1e-12, None),
        ) / math.log(2)
        jsd_parts.append(0.5 * (observed_term + predicted_term).sum(axis=1))
        mae_parts.append(
            np.abs(observed_probability - predicted_probability).mean(axis=1)
        )

    if jsd_parts:
        jsd_values = np.concatenate(jsd_parts).astype(np.float64)
        mae_values = np.concatenate(mae_parts).astype(np.float64)
        macro_jsd = float(jsd_values.mean())
        macro_mae = float(mae_values.mean())
        n_evaluable = int(len(jsd_values))
    else:
        macro_jsd = float("nan")
        macro_mae = float("nan")
        n_evaluable = 0

    metrics = {
        "allocation_nll": nll,
        "allocation_perplexity": float(math.exp(min(nll, 30))),
        "macro_cell_gene_jsd": macro_jsd,
        "cell_gene_proportion_mae": macro_mae,
        "n_evaluable_cell_genes": n_evaluable,
        "n_cells": int(y.shape[0]),
        "n_isoforms": int(y.shape[1]),
        "observed_umis": int(total),
    }
    metrics.update(
        evaluate_confident_dominant_accuracy(
            y,
            probabilities,
            group_index,
            min_parent_umis=min_gene_umis,
        )
    )
    return metrics


def evaluate_cluster_proportion_spearman(
    true_counts: sparse.csr_matrix | np.ndarray,
    probabilities: np.ndarray,
    group_index: np.ndarray,
    cluster_labels: np.ndarray,
    min_parent_umis: int = 2,
) -> dict[str, float | list[float] | list[int]]:
    """Compare unweighted mean isoform fractions inside expression clusters."""
    y = true_counts.toarray() if sparse.issparse(true_counts) else np.asarray(true_counts)
    y = y.astype(np.float32, copy=False)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    group_index = np.asarray(group_index, dtype=np.int64)
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    n_groups = int(group_index.max()) + 1
    gene_totals = np.zeros((len(y), n_groups), dtype=np.float32)
    np.add.at(gene_totals.T, group_index, y.T)
    observed_by_cluster: dict[int, list[np.ndarray]] = {}
    predicted_by_cluster: dict[int, list[np.ndarray]] = {}

    for group in range(n_groups):
        ix = np.flatnonzero(group_index == group)
        if len(ix) < 2:
            continue
        valid_coverage = gene_totals[:, group] >= min_parent_umis
        for cluster in np.unique(cluster_labels):
            valid = valid_coverage & (cluster_labels == cluster)
            if not valid.any():
                continue
            observed = y[valid][:, ix]
            observed_fraction = observed / observed.sum(axis=1, keepdims=True)
            predicted_fraction = probabilities[valid][:, ix]
            observed_by_cluster.setdefault(int(cluster), []).append(
                observed_fraction.mean(axis=0)
            )
            predicted_by_cluster.setdefault(int(cluster), []).append(
                predicted_fraction.mean(axis=0)
            )

    values: list[float] = []
    sizes: list[int] = []
    cluster_ids: list[int] = []
    for cluster in sorted(observed_by_cluster):
        observed = np.concatenate(observed_by_cluster[cluster])
        predicted = np.concatenate(predicted_by_cluster[cluster])
        correlation = _safe_spearman(observed, predicted)
        if np.isfinite(correlation):
            values.append(float(correlation))
            sizes.append(int(len(observed)))
            cluster_ids.append(cluster)
    array = np.asarray(values, dtype=np.float64)
    return {
        "cluster_allocation_spearman_median": float(np.median(array)) if len(array) else float("nan"),
        "cluster_allocation_spearman_mean": float(np.mean(array)) if len(array) else float("nan"),
        "cluster_allocation_spearman_min": float(np.min(array)) if len(array) else float("nan"),
        "cluster_allocation_spearman_max": float(np.max(array)) if len(array) else float("nan"),
        "n_clusters_allocation_spearman": int(len(array)),
        "median_isoforms_per_cluster_spearman": float(np.median(sizes)) if sizes else float("nan"),
        "cluster_allocation_spearman_values": values,
        "cluster_allocation_spearman_cluster_ids": cluster_ids,
    }


def evaluate_spatial_proportion_moran(
    true_counts: sparse.csr_matrix | np.ndarray,
    probabilities: np.ndarray,
    group_index: np.ndarray,
    coordinates: np.ndarray,
    k: int = 6,
    max_isoforms: int = 250,
    min_parent_umis: int = 2,
    min_valid_spots: int = 20,
) -> float:
    """Spearman agreement of observed and predicted fraction-based Moran's I."""
    y = true_counts.toarray() if sparse.issparse(true_counts) else np.asarray(true_counts)
    y = y.astype(np.float32, copy=False)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    group_index = np.asarray(group_index, dtype=np.int64)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.shape != (len(y), 2):
        return float("nan")

    n_groups = int(group_index.max()) + 1
    gene_totals = np.zeros((len(y), n_groups), dtype=np.float32)
    np.add.at(gene_totals.T, group_index, y.T)
    group_sizes = np.bincount(group_index, minlength=n_groups)
    candidates = np.flatnonzero(group_sizes[group_index] >= 2)
    if len(candidates) > max_isoforms:
        bulk = y[:, candidates].sum(axis=0)
        candidates = candidates[np.argpartition(bulk, -max_isoforms)[-max_isoforms:]]

    def moran(values: np.ndarray, neighbors: np.ndarray) -> float:
        if float(np.ptp(values)) <= 1e-7:
            return 0.0
        centered = values - values.mean()
        denominator = float(np.square(centered).sum())
        if denominator <= 1e-12:
            return 0.0
        numerator = float((centered[:, None] * centered[neighbors]).sum())
        return len(values) / max(neighbors.size, 1) * numerator / denominator

    observed_moran: list[float] = []
    predicted_moran: list[float] = []
    required = max(min_valid_spots, k + 2)
    for isoform in candidates:
        group = group_index[isoform]
        valid = gene_totals[:, group] >= min_parent_umis
        if int(valid.sum()) < required:
            continue
        valid_coordinates = coordinates[valid]
        neighbor_count = min(k + 1, int(valid.sum()))
        _, neighbors = cKDTree(valid_coordinates).query(
            valid_coordinates, k=neighbor_count
        )
        neighbors = neighbors[:, 1:]
        observed_fraction = y[valid, isoform] / gene_totals[valid, group]
        predicted_fraction = probabilities[valid, isoform]
        observed_moran.append(moran(observed_fraction, neighbors))
        predicted_moran.append(moran(predicted_fraction, neighbors))
    return _safe_spearman(
        np.asarray(observed_moran, dtype=np.float64),
        np.asarray(predicted_moran, dtype=np.float64),
    )


def evaluate_spatial_moran_matrix_only(
    true_counts: sparse.csr_matrix | np.ndarray,
    probabilities: np.ndarray,
    group_index: np.ndarray,
    coordinates: np.ndarray,
    predicted_parent_abundance: sparse.csr_matrix | np.ndarray,
    k: int = 6,
    max_isoforms: int = 250,
) -> float:
    """Spatial autocorrelation agreement with strictly short-read predictions."""
    y = true_counts.toarray() if sparse.issparse(true_counts) else np.asarray(true_counts)
    y = y.astype(np.float32, copy=False)
    parent = (
        predicted_parent_abundance.toarray()
        if sparse.issparse(predicted_parent_abundance)
        else np.asarray(predicted_parent_abundance)
    ).astype(np.float32, copy=False)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if len(y) < k + 2 or coordinates.shape != (len(y), 2):
        return float("nan")
    predicted = probabilities * np.maximum(parent, 0)[:, group_index]
    group_sizes = np.bincount(group_index)
    candidates = np.flatnonzero(group_sizes[group_index] >= 2)
    if len(candidates) > max_isoforms:
        # The evaluation subset is defined once from observed abundance and is
        # therefore identical for every benchmark method.
        bulk = y[:, candidates].sum(axis=0)
        candidates = candidates[np.argpartition(bulk, -max_isoforms)[-max_isoforms:]]
    truth = y[:, candidates]
    estimate = predicted[:, candidates]
    _, neighbors = cKDTree(coordinates).query(coordinates, k=min(k + 1, len(y)))
    neighbors = neighbors[:, 1:]

    def moran(matrix: np.ndarray) -> np.ndarray:
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        denominator = np.square(centered).sum(axis=0)
        numerator = (centered[:, None, :] * centered[neighbors]).sum(axis=(0, 1))
        scale = len(matrix) / max(neighbors.size, 1)
        return scale * numerator / np.maximum(denominator, 1e-8)

    return _safe_spearman(moran(truth), moran(estimate))


def _columnwise_spearman(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Spearman correlation for corresponding columns without a Python loop."""
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("columnwise Spearman requires equally shaped 2D matrices")
    left_rank = stats.rankdata(left, axis=0).astype(np.float64)
    right_rank = stats.rankdata(right, axis=0).astype(np.float64)
    left_rank -= left_rank.mean(axis=0, keepdims=True)
    right_rank -= right_rank.mean(axis=0, keepdims=True)
    denominator = np.sqrt(
        np.square(left_rank).sum(axis=0) * np.square(right_rank).sum(axis=0)
    )
    correlations = np.full(left.shape[1], np.nan, dtype=np.float64)
    left_range = np.ptp(left, axis=0)
    right_range = np.ptp(right, axis=0)
    left_scale = np.maximum(np.max(np.abs(left), axis=0), 1.0)
    right_scale = np.maximum(np.max(np.abs(right), axis=0), 1.0)
    valid = (
        (denominator > 0)
        & (left_range > 1e-5 * left_scale)
        & (right_range > 1e-5 * right_scale)
    )
    correlations[valid] = (
        (left_rank[:, valid] * right_rank[:, valid]).sum(axis=0)
        / denominator[valid]
    )
    return correlations


def evaluate_isoform_cell_spearman(
    true_counts: sparse.csr_matrix | np.ndarray,
    probabilities: np.ndarray,
    group_index: np.ndarray,
    min_parent_umis: int = 2,
    min_cells: int = 30,
    min_detected_cells: int = 5,
    smoothing: float = 0.5,
) -> dict[str, float]:
    """Macro isoform-wise correlation of cell-to-cell usage variation."""
    y = true_counts.toarray() if sparse.issparse(true_counts) else np.asarray(true_counts)
    y = y.astype(np.float32, copy=False)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    group_index = np.asarray(group_index, dtype=np.int64)
    n_groups = int(group_index.max()) + 1
    gene_totals = np.zeros((len(y), n_groups), dtype=np.float32)
    np.add.at(gene_totals.T, group_index, y.T)
    correlations: list[np.ndarray] = []
    for group in range(n_groups):
        ix = np.flatnonzero(group_index == group)
        valid_cells = gene_totals[:, group] >= min_parent_umis
        if len(ix) < 2 or valid_cells.sum() < min_cells:
            continue
        observed = y[valid_cells][:, ix]
        detected = (observed > 0).sum(axis=0) >= min_detected_cells
        if not detected.any():
            continue
        observed_probability = (observed + smoothing) / (
            observed.sum(axis=1, keepdims=True) + smoothing * len(ix)
        )
        group_correlations = _columnwise_spearman(
            observed_probability[:, detected],
            probabilities[valid_cells][:, ix][:, detected],
        )
        correlations.append(group_correlations[np.isfinite(group_correlations)])
    values = np.concatenate(correlations) if correlations else np.asarray([], dtype=float)
    return {
        "isoform_cell_spearman_median": float(np.median(values)) if len(values) else float("nan"),
        "isoform_cell_spearman_mean": float(np.mean(values)) if len(values) else float("nan"),
        "n_isoforms_cell_spearman": int(len(values)),
    }


def _aggregate_rows(
    matrix: sparse.csr_matrix | np.ndarray, labels: np.ndarray
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    n_groups = int(labels.max()) + 1
    aggregation = sparse.csr_matrix(
        (
            np.ones(len(labels), dtype=np.float32),
            (labels, np.arange(len(labels))),
        ),
        shape=(n_groups, len(labels)),
    )
    result = aggregation @ matrix
    return result.toarray() if sparse.issparse(result) else np.asarray(result)


def evaluate_pseudocell_isoform_spearman(
    true_counts: sparse.csr_matrix | np.ndarray,
    probabilities: np.ndarray,
    group_index: np.ndarray,
    predicted_parent_abundance: sparse.csr_matrix | np.ndarray,
    pseudocell_labels: np.ndarray,
    min_parent_umis: int = 10,
    min_pseudocells: int = 8,
    min_detected_pseudocells: int = 3,
    smoothing: float = 0.5,
) -> dict[str, float]:
    """Isoform-wise usage correlation across short-read-defined pseudocells."""
    y_grouped = _aggregate_rows(true_counts, pseudocell_labels).astype(np.float32)
    parent = (
        predicted_parent_abundance.toarray()
        if sparse.issparse(predicted_parent_abundance)
        else np.asarray(predicted_parent_abundance)
    ).astype(np.float32, copy=False)
    predicted_counts = probabilities * np.maximum(parent, 0)[:, group_index]
    predicted_grouped = _aggregate_rows(predicted_counts, pseudocell_labels).astype(np.float32)
    correlations: list[np.ndarray] = []
    for group in range(int(group_index.max()) + 1):
        ix = np.flatnonzero(group_index == group)
        true_total = y_grouped[:, ix].sum(axis=1)
        predicted_total = predicted_grouped[:, ix].sum(axis=1)
        valid = (true_total >= min_parent_umis) & (predicted_total > 0)
        if len(ix) < 2 or valid.sum() < min_pseudocells:
            continue
        observed = y_grouped[valid][:, ix]
        detected = (observed > 0).sum(axis=0) >= min_detected_pseudocells
        if not detected.any():
            continue
        observed_probability = (observed + smoothing) / (
            observed.sum(axis=1, keepdims=True) + smoothing * len(ix)
        )
        predicted_probability = predicted_grouped[valid][:, ix]
        predicted_probability /= np.maximum(
            predicted_probability.sum(axis=1, keepdims=True), 1e-12
        )
        group_correlations = _columnwise_spearman(
            observed_probability[:, detected], predicted_probability[:, detected]
        )
        correlations.append(group_correlations[np.isfinite(group_correlations)])
    values = np.concatenate(correlations) if correlations else np.asarray([], dtype=float)
    return {
        "isoform_pseudocell_spearman_median": float(np.median(values)) if len(values) else float("nan"),
        "isoform_pseudocell_spearman_mean": float(np.mean(values)) if len(values) else float("nan"),
        "n_isoforms_pseudocell_spearman": int(len(values)),
        "n_pseudocells": int(np.max(pseudocell_labels) + 1),
    }


def evaluate_cluster_usage_and_switches(
    true_counts: sparse.csr_matrix | np.ndarray,
    probabilities: np.ndarray,
    group_index: np.ndarray,
    predicted_parent_abundance: sparse.csr_matrix | np.ndarray,
    cluster_labels: np.ndarray,
    min_parent_umis: int = 10,
    min_dominant_fraction: float = 0.5,
    min_dominant_margin: float = 0.1,
) -> dict[str, float]:
    """Cluster-level composition error and confident dominant-isoform switches."""
    y_grouped = _aggregate_rows(true_counts, cluster_labels).astype(np.float32)
    parent = (
        predicted_parent_abundance.toarray()
        if sparse.issparse(predicted_parent_abundance)
        else np.asarray(predicted_parent_abundance)
    ).astype(np.float32, copy=False)
    predicted_counts = probabilities * np.maximum(parent, 0)[:, group_index]
    predicted_grouped = _aggregate_rows(predicted_counts, cluster_labels).astype(np.float32)
    jsd_values: list[float] = []
    jsd_weights: list[float] = []
    truth_switches: list[bool] = []
    predicted_switches: list[bool] = []
    exact_switches: list[bool] = []
    cluster_observed_probabilities: list[list[np.ndarray]] = [
        [] for _ in range(y_grouped.shape[0])
    ]
    cluster_predicted_probabilities: list[list[np.ndarray]] = [
        [] for _ in range(y_grouped.shape[0])
    ]
    for group in range(int(group_index.max()) + 1):
        ix = np.flatnonzero(group_index == group)
        if len(ix) < 2:
            continue
        observed = y_grouped[:, ix]
        predicted = predicted_grouped[:, ix]
        observed_total = observed.sum(axis=1)
        predicted_total = predicted.sum(axis=1)
        valid = (observed_total >= min_parent_umis) & (predicted_total > 0)
        if not valid.any():
            continue
        observed_probability = observed[valid] / observed_total[valid, None]
        predicted_probability = predicted[valid] / predicted_total[valid, None]
        for local_row, cluster in enumerate(np.flatnonzero(valid)):
            cluster_observed_probabilities[cluster].append(
                observed_probability[local_row]
            )
            cluster_predicted_probabilities[cluster].append(
                predicted_probability[local_row]
            )
        midpoint = 0.5 * (observed_probability + predicted_probability)
        jsd = 0.5 * (
            special.xlogy(
                observed_probability,
                observed_probability / np.clip(midpoint, 1e-12, None),
            ).sum(axis=1)
            + special.xlogy(
                predicted_probability,
                predicted_probability / np.clip(midpoint, 1e-12, None),
            ).sum(axis=1)
        ) / math.log(2)
        jsd_values.extend(jsd.tolist())
        jsd_weights.extend(observed_total[valid].tolist())

        observed_order = np.argsort(observed_probability, axis=1)
        observed_dominant = observed_order[:, -1]
        predicted_dominant = np.argmax(predicted_probability, axis=1)
        top = np.take_along_axis(
            observed_probability, observed_dominant[:, None], axis=1
        ).ravel()
        second = np.take_along_axis(
            observed_probability, observed_order[:, -2, None], axis=1
        ).ravel()
        confident = (top >= min_dominant_fraction) & (
            (top - second) >= min_dominant_margin
        )
        confident_indices = np.flatnonzero(confident)
        for left_position, left in enumerate(confident_indices):
            for right in confident_indices[left_position + 1 :]:
                truth_switch = observed_dominant[left] != observed_dominant[right]
                predicted_switch = predicted_dominant[left] != predicted_dominant[right]
                truth_switches.append(bool(truth_switch))
                predicted_switches.append(bool(predicted_switch))
                if truth_switch:
                    exact_switches.append(
                        bool(
                            predicted_dominant[left] == observed_dominant[left]
                            and predicted_dominant[right] == observed_dominant[right]
                        )
                    )

    jsd_array = np.asarray(jsd_values, dtype=np.float64)
    weight_array = np.asarray(jsd_weights, dtype=np.float64)
    truth_array = np.asarray(truth_switches, dtype=bool)
    predicted_array = np.asarray(predicted_switches, dtype=bool)
    cluster_allocation_correlations: list[float] = []
    cluster_allocation_sizes: list[int] = []
    cluster_allocation_ids: list[int] = []
    for cluster_id, (observed_parts, predicted_parts) in enumerate(zip(
        cluster_observed_probabilities, cluster_predicted_probabilities
    )):
        if not observed_parts:
            continue
        observed_values = np.concatenate(observed_parts)
        predicted_values = np.concatenate(predicted_parts)
        correlation = _safe_spearman(observed_values, predicted_values)
        if np.isfinite(correlation):
            cluster_allocation_correlations.append(correlation)
            cluster_allocation_sizes.append(int(len(observed_values)))
            cluster_allocation_ids.append(int(cluster_id))
    allocation_array = np.asarray(cluster_allocation_correlations, dtype=np.float64)
    if len(truth_array):
        true_positive = int(np.sum(truth_array & predicted_array))
        false_positive = int(np.sum(~truth_array & predicted_array))
        false_negative = int(np.sum(truth_array & ~predicted_array))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        switch_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if len(np.unique(truth_array)) < 2 or len(np.unique(predicted_array)) < 2:
            switch_mcc = 0.0
        else:
            switch_mcc = float(matthews_corrcoef(truth_array, predicted_array))
    else:
        precision = recall = switch_f1 = switch_mcc = float("nan")
    metrics = {
        "cluster_gene_jsd_macro": float(jsd_array.mean()) if len(jsd_array) else float("nan"),
        "cluster_gene_jsd_weighted": float(np.average(jsd_array, weights=weight_array)) if len(jsd_array) else float("nan"),
        "n_evaluable_cluster_genes": int(len(jsd_array)),
        "cluster_allocation_spearman_median": float(np.median(allocation_array)) if len(allocation_array) else float("nan"),
        "cluster_allocation_spearman_mean": float(np.mean(allocation_array)) if len(allocation_array) else float("nan"),
        "cluster_allocation_spearman_min": float(np.min(allocation_array)) if len(allocation_array) else float("nan"),
        "cluster_allocation_spearman_max": float(np.max(allocation_array)) if len(allocation_array) else float("nan"),
        "n_clusters_allocation_spearman": int(len(allocation_array)),
        "median_isoforms_per_cluster_spearman": float(np.median(cluster_allocation_sizes)) if cluster_allocation_sizes else float("nan"),
        "cluster_allocation_spearman_values": [
            float(value) for value in cluster_allocation_correlations
        ],
        "cluster_allocation_spearman_cluster_ids": cluster_allocation_ids,
        "dominant_switch_precision": float(precision),
        "dominant_switch_recall": float(recall),
        "dominant_switch_f1": float(switch_f1),
        "dominant_switch_mcc": float(switch_mcc),
        "dominant_switch_exact_accuracy": float(np.mean(exact_switches)) if exact_switches else float("nan"),
        "n_cluster_pairs_for_switch": int(len(truth_array)),
        "n_true_dominant_switches": int(truth_array.sum()),
    }
    cluster_dominant = evaluate_confident_dominant_accuracy(
        y_grouped,
        predicted_grouped,
        group_index,
        min_parent_umis=min_parent_umis,
        min_dominant_fraction=min_dominant_fraction,
        min_dominant_margin=min_dominant_margin,
    )
    metrics.update(
        {f"cluster_{key}": value for key, value in cluster_dominant.items()}
    )
    return metrics


def evaluate_spatial_moran(
    true_counts: sparse.csr_matrix | np.ndarray,
    probabilities: np.ndarray,
    group_index: np.ndarray,
    coordinates: np.ndarray,
    parent_abundance: sparse.csr_matrix | np.ndarray | None = None,
    k: int = 6,
    max_isoforms: int = 250,
) -> float:
    """Correlation of observed and predicted Moran's I on one held-out section."""
    y = true_counts.toarray() if sparse.issparse(true_counts) else np.asarray(true_counts)
    y = y.astype(np.float32, copy=False)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if len(y) < k + 2 or coordinates.shape != (len(y), 2):
        return float("nan")
    gene_totals = np.zeros((y.shape[0], int(group_index.max()) + 1), dtype=np.float32)
    np.add.at(gene_totals.T, group_index, y.T)
    if parent_abundance is None:
        scaled_parent = gene_totals
    else:
        scaled_parent = (
            parent_abundance.toarray()
            if sparse.issparse(parent_abundance)
            else np.asarray(parent_abundance)
        ).astype(np.float32, copy=False)
        short_total = scaled_parent.sum(axis=1)
        scale = y.sum(axis=1) / np.maximum(short_total, 1e-8)
        scaled_parent = np.maximum(scaled_parent, 0) * scale[:, None]
    pred = probabilities * scaled_parent[:, group_index]
    group_sizes = np.bincount(group_index)
    candidates = np.flatnonzero(group_sizes[group_index] >= 2)
    if len(candidates) > max_isoforms:
        bulk = y[:, candidates].sum(axis=0)
        candidates = candidates[np.argpartition(bulk, -max_isoforms)[-max_isoforms:]]
    truth = y[:, candidates]
    estimate = pred[:, candidates]
    _, neighbors = cKDTree(coordinates).query(coordinates, k=min(k + 1, len(y)))
    neighbors = neighbors[:, 1:]

    def moran(matrix: np.ndarray) -> np.ndarray:
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        denominator = np.square(centered).sum(axis=0)
        numerator = (centered[:, None, :] * centered[neighbors]).sum(axis=(0, 1))
        scale = len(matrix) / max(neighbors.size, 1)
        return scale * numerator / np.maximum(denominator, 1e-8)

    observed_i = moran(truth)
    predicted_i = moran(estimate)
    return _safe_spearman(observed_i, predicted_i)


def quality_gate(metrics: dict[str, float], spatial_moran_spearman: float | None = None) -> dict:
    """Absolute, baseline-free feasibility criteria.

    The thresholds are deliberately moderate for sparse single-cell long reads.
    A dataset is retained only if all core criteria with enough observations pass.
    """
    checks = {
        "enough_validation_cells": metrics.get("n_cells", 0) >= 100,
        "enough_long_read_umis": metrics.get("observed_umis", 0) >= 1000,
        "allocation_pseudobulk_spearman>=0.60": metrics.get(
            "allocation_pseudobulk_isoform_spearman", -1
        ) >= 0.60,
        "end_to_end_pseudobulk_spearman>=0.45": metrics.get(
            "pseudobulk_isoform_spearman", -1
        ) >= 0.45,
        "weighted_jsd<=0.30": metrics.get("weighted_gene_jsd", 1) <= 0.30,
        "dominant_accuracy>=0.50": metrics.get("dominant_isoform_accuracy", -1) >= 0.50,
        "auprc_lift>=0.05": metrics.get("detection_auprc_lift", -1) >= 0.05,
    }
    if spatial_moran_spearman is not None:
        checks["spatial_moran_spearman>=0.50"] = spatial_moran_spearman >= 0.50
    return {"passed": all(checks.values()), "checks": checks}
