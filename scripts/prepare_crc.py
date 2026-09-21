#!/usr/bin/env python3
"""Prepare matched Illumina/PacBio colorectal cancer cells."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from short2long.data import PairedDataset, save_dataset  # noqa: E402
from short2long.structure import structures_from_annotation  # noqa: E402


def load_isoform_gene_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "transcript":
                continue
            attributes = fields[8]
            tx = re.search(r"(?:^|; )ID=([^;]+)", attributes)
            gene = re.search(r"(?:^|; )gene_name=([^;]+)", attributes)
            if tx and gene:
                mapping[tx.group(1)] = gene.group(1)
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/crc_cross_donor"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/crc_paired"))
    args = parser.parse_args()

    short_path = args.raw_root / "10XIllumina.combined.TPM.h5ad"
    long_path = args.raw_root / "PacBio-isoform_counts_matrix_ad.h5ad"
    gff_path = (
        args.raw_root
        / "extracted"
        / "gffcmp.query.all_samples.combined.colored.sorted.gff3"
    )
    short = ad.read_h5ad(short_path, backed="r")
    long = ad.read_h5ad(long_path)
    common = short.obs_names[short.obs_names.isin(long.obs_names)]
    short_positions = short.obs_names.get_indexer(common)
    long_positions = long.obs_names.get_indexer(common)
    x = short.X[short_positions].tocsr().astype(np.float32)
    # This public matrix stores log1p-normalized expression.  Inverting it to
    # linear CPM lets the common normalization path recover the supplied values.
    x.data = np.expm1(x.data)
    y = long.X[long_positions].tocsr().astype(np.float32)

    covered = np.asarray(y.sum(axis=1)).ravel() > 0
    common = common[covered]
    x = x[covered]
    y = y[covered]
    long_positions = long_positions[covered]

    isoform_to_gene = load_isoform_gene_map(gff_path)
    isoforms = long.var_names.astype(str).to_numpy()
    isoform_genes = np.asarray([isoform_to_gene.get(tx, "") for tx in isoforms])
    short_genes = short.var_names.astype(str).to_numpy()
    keep_isoform = np.isin(isoform_genes, short_genes) & (isoform_genes != "")
    y = y[:, keep_isoform]
    isoforms = isoforms[keep_isoform]
    isoform_genes = isoform_genes[keep_isoform]
    isoform_structures, recovered_structures = structures_from_annotation(
        gff_path, isoforms, isoform_genes
    )

    source_obs = short.obs.loc[common]
    obs = pd.DataFrame(index=common.astype(str))
    obs["sample"] = source_obs["library_id"].astype(str).to_numpy()
    obs["donor"] = obs["sample"].str.extract(r"(PS\d+)", expand=False)
    obs["condition"] = source_obs["condition"].astype(str).to_numpy()
    obs["cell_type"] = source_obs["ClusterMidway"].astype(str).to_numpy()
    obs["short_umis"] = source_obs["nCount_RNA"].to_numpy(dtype=float)
    obs["long_reads"] = np.asarray(y.sum(axis=1)).ravel()
    if obs["donor"].isna().any() or obs["donor"].nunique() < 3:
        raise ValueError("could not recover at least three donors")

    dataset = PairedDataset(
        x=x,
        y=y,
        obs=obs,
        genes=short_genes,
        isoforms=isoforms,
        isoform_genes=isoform_genes,
        isoform_structures=isoform_structures,
        metadata={
            "categories": ["cross_donor", "cross_disease"],
            "study": "colorectal cancer long-read single-cell atlas",
            "input": "matched 10x Illumina log-normalized gene expression",
            "target": "matched PacBio isoform counts",
            "pairing": "same 10x cell barcode",
            "donors": sorted(obs["donor"].unique().tolist()),
            "conditions": sorted(obs["condition"].unique().tolist()),
            "recommended_donor_holdout": "PS018",
            "recommended_disease_holdout": "tumor",
            "isoform_structures_recovered": recovered_structures,
        },
    )
    save_dataset(dataset, args.output)
    short.file.close()
    print(
        f"saved {dataset.x.shape[0]} cells, {dataset.x.shape[1]} genes, "
        f"{dataset.y.shape[1]} mapped isoforms, {obs['donor'].nunique()} donors to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
