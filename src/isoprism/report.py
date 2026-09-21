"""Export run-level scores, seed summaries and every defined cluster point."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def export_tables(root):
    root = Path(root)
    runs, points = [], []
    for path in sorted(root.glob("*/*/*/result.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("smoke"):
            continue
        metrics = record["metrics"]
        method = record.get("method", record.get("variant"))
        if method == "Competitive-TA":
            method = "IsoPrism"
        rhos = metrics.get("cluster_allocation_spearman_values", [])
        ids = metrics.get("cluster_allocation_spearman_cluster_ids", [])
        if len(rhos) != len(ids):
            raise ValueError("Mismatched cluster identifiers and Spearman values")
        row = {"setting": record["task"], "method": method, "seed": record.get("seed"),
               "dominant_isoform_accuracy": metrics.get("confident_dominant_isoform_gene_macro_accuracy"),
               "cell_gene_1_minus_jsd": metrics.get("cell_gene_1_minus_jsd"),
               "cluster_spearman_max": max(rhos) if rhos else None,
               "cluster_spearman_median": metrics.get("cluster_allocation_spearman_median"),
               "spatial_moran_spearman": metrics.get("spatial_proportion_moran_spearman"),
               "confident_genes": metrics.get("n_genes_confident_dominant"),
               "eligible_cell_gene_pairs": metrics.get("n_evaluable_cell_genes"),
               "defined_clusters": len(rhos)}
        runs.append(row)
        for cluster, rho in zip(ids, rhos):
            points.append({"setting": row["setting"], "method": method, "seed": row["seed"],
                           "cluster_id": cluster, "allocation_spearman": rho})
    if not runs:
        return
    table = pd.DataFrame(runs)
    table.to_csv(root / "individual_runs.csv", index=False, na_rep="NA")
    pd.DataFrame(points, columns=["setting", "method", "seed", "cluster_id", "allocation_spearman"]).to_csv(
        root / "cluster_points.csv", index=False, na_rep="NA")
    fields = ["dominant_isoform_accuracy", "cell_gene_1_minus_jsd", "cluster_spearman_max",
              "cluster_spearman_median", "spatial_moran_spearman"]
    summary = []
    for (setting, method), group in table.groupby(["setting", "method"], sort=False):
        for metric in fields:
            valid = pd.to_numeric(group[metric], errors="coerce").dropna()
            summary.append({"setting": setting, "method": method, "metric": metric,
                            "mean": valid.mean() if len(valid) else np.nan,
                            "sample_sd": valid.std(ddof=1) if len(valid) > 1 else np.nan,
                            "defined_runs": len(valid), "completed_runs": len(group)})
    pd.DataFrame(summary).to_csv(root / "summary.csv", index=False, na_rep="NA")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    export_tables(parser.parse_args().root)


if __name__ == "__main__":
    main()
