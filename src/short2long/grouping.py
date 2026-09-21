from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .data import normalize_log_cpm


@dataclass(frozen=True)
class PseudoclusterBundle:
    cell_context: np.ndarray
    group_expression: np.ndarray
    group_counts: np.ndarray
    group_labels: np.ndarray
    coarse_labels: np.ndarray
    sizes: np.ndarray


def group_means(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    n_groups = int(labels.max()) + 1
    result = np.zeros((n_groups, values.shape[1]), dtype=np.float32)
    np.add.at(result, labels, values)
    sizes = np.bincount(labels, minlength=n_groups).astype(np.float32)
    result /= np.maximum(sizes[:, None], 1.0)
    return result


def repeated_group_means(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    means = group_means(values, labels)
    return means[np.asarray(labels, dtype=np.int64)]


def _strata(values: np.ndarray | None, n_cells: int) -> np.ndarray:
    if values is None:
        return np.repeat("all", n_cells)
    result = np.asarray(values, dtype=object).astype(str)
    if result.shape != (n_cells,):
        raise ValueError("strata must contain one value per cell")
    result[np.isin(result, ["", "nan", "None", "<NA>"])] = "unknown"
    return result


def build_pseudocluster_labels(
    expression: np.ndarray,
    seed: int,
    target_size: int = 50,
    requested_clusters: int = 10,
    pca_components: int = 32,
    strata: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Define local pseudoclusters using short-read expression only."""
    expression = np.asarray(expression, dtype=np.float32)
    if expression.ndim != 2 or len(expression) == 0:
        raise ValueError("expression must be a non-empty cell-by-gene matrix")
    components = min(pca_components, len(expression) - 1, expression.shape[1])
    if components >= 1:
        embedding = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=seed,
        ).fit_transform(expression).astype(np.float32)
    else:
        embedding = expression[:, :1]

    strata_values = _strata(strata, len(expression))
    pseudocluster = np.full(len(expression), -1, dtype=np.int64)
    coarse = np.full(len(expression), -1, dtype=np.int64)
    next_pseudo = 0
    next_coarse = 0
    for stratum in sorted(np.unique(strata_values)):
        members = np.flatnonzero(strata_values == stratum)
        n_coarse = min(requested_clusters, max(1, len(members) // 100))
        if n_coarse == 1:
            local_coarse = np.zeros(len(members), dtype=np.int64)
        else:
            local_coarse = KMeans(
                n_clusters=n_coarse, n_init=20, random_state=seed
            ).fit_predict(embedding[members])
        for local_cluster in range(n_coarse):
            local_members = members[local_coarse == local_cluster]
            if not len(local_members):
                continue
            coarse[local_members] = next_coarse
            next_coarse += 1
            order = local_members[np.argsort(embedding[local_members, 0])]
            n_pseudo = max(1, int(round(len(order) / target_size)))
            for chunk in np.array_split(order, n_pseudo):
                pseudocluster[chunk] = next_pseudo
                next_pseudo += 1
    if np.any(pseudocluster < 0) or np.any(coarse < 0):
        raise RuntimeError("not every cell was assigned to a pseudocluster")
    return pseudocluster, coarse


def build_pseudocluster_bundle(
    raw_expression: sparse.csr_matrix,
    normalized_expression: np.ndarray,
    isoform_counts: np.ndarray,
    seed: int,
    target_size: int = 50,
    requested_clusters: int = 10,
    pca_components: int = 32,
    strata: np.ndarray | None = None,
) -> PseudoclusterBundle:
    labels, coarse = build_pseudocluster_labels(
        normalized_expression,
        seed=seed,
        target_size=target_size,
        requested_clusters=requested_clusters,
        pca_components=pca_components,
        strata=strata,
    )
    n_groups = int(labels.max()) + 1
    membership = sparse.csr_matrix(
        (
            np.ones(len(labels), dtype=np.float32),
            (labels, np.arange(len(labels), dtype=np.int64)),
        ),
        shape=(n_groups, len(labels)),
    )
    group_raw = (membership @ raw_expression).tocsr()
    group_expression = normalize_log_cpm(group_raw)
    group_counts = np.asarray(membership @ isoform_counts, dtype=np.float32)
    cell_context = group_expression[labels]
    sizes = np.bincount(labels, minlength=n_groups).astype(np.int64)
    return PseudoclusterBundle(
        cell_context=cell_context,
        group_expression=group_expression,
        group_counts=group_counts,
        group_labels=labels,
        coarse_labels=coarse,
        sizes=sizes,
    )
