"""Shared fixed-mask proportion metrics, including honest constant-map handling."""
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from .coverage_retrain import evaluate_sparse
from .metrics import _safe_spearman


def moran(values, neighbors):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.ptp(values) <= 1e-7:
        return np.nan
    centered = values - values.mean()
    return len(values)/neighbors.size * np.sum(centered[:,None]*centered[neighbors])/np.sum(centered**2)


def spatial_metrics(y, p, groups, coordinates, panel):
    y = sparse.csc_matrix(y)
    records = []
    for gene in np.unique(groups[np.asarray(panel)]):
        ix = np.flatnonzero(groups == gene)
        counts = y[:,ix].toarray()
        totals = counts.sum(1)
        valid = totals >= 2
        if valid.sum() < 20:
            continue
        coords = np.asarray(coordinates[valid])
        if not np.isfinite(coords).all():
            raise ValueError("Non-finite spatial coordinates")
        nearest = cKDTree(coords).query(coords, k=7)[1]
        # Explicitly remove self even when duplicate coordinates reorder ties.
        neighbors = np.asarray([row[row != j][:6] for j,row in enumerate(nearest)])
        for local, iso in enumerate(ix):
            if iso not in panel:
                continue
            observed = moran(counts[valid,local]/totals[valid], neighbors)
            if not np.isfinite(observed):
                continue
            predicted = moran(np.asarray(p[valid,iso]), neighbors)
            records.append({"isoform_index": int(iso), "observed_moran": observed,
                            "predicted_moran": predicted, "n_spots": int(valid.sum())})
    valid_records = [r for r in records if np.isfinite(r["predicted_moran"])]
    rho = _safe_spearman(np.asarray([r["observed_moran"] for r in valid_records]),
                         np.asarray([r["predicted_moran"] for r in valid_records])) if len(valid_records) >= 3 else np.nan
    return {"spatial_proportion_moran_spearman": rho if len(valid_records) == len(records) else None,
            "spatial_pairwise_valid_spearman_diagnostic": rho, "spatial_source_panel_size": len(panel),
            "spatial_truth_panel_size": len(records), "spatial_prediction_valid_size": len(valid_records),
            "spatial_constant_or_invalid_predictions": len(records)-len(valid_records), "spatial_points": records}


def evaluate(view, probabilities):
    truth = sparse.load_npz(view["folder"]/"test_truth.npz")
    result = evaluate_sparse(truth, probabilities, view["groups"], view["evaluation_clusters"])
    result["cell_gene_1_minus_jsd"] = 1-result["macro_cell_gene_jsd"] if result["macro_cell_gene_jsd"] is not None else None
    if "coordinates" in view:
        result.update(spatial_metrics(truth, probabilities, view["groups"], view["coordinates"], view["spatial_panel"]))
    return result
