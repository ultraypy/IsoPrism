#!/usr/bin/env python3
"""Prepare matched Illumina/Nanopore Visium spots from the post-MI study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from short2long.data import PairedDataset, save_dataset  # noqa: E402
from short2long.structure import structures_from_annotation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/raw/post_mi_spatial/extracted")
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/post_mi_spatial"))
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("data/raw/post_mi_spatial/gffcmp.multi_exons.annotated.gtf"),
    )
    args = parser.parse_args()

    short = ad.read_h5ad(args.raw_root / "illumina.h5ad")
    long = ad.read_h5ad(args.raw_root / "nanopore.h5ad")
    common = short.obs_names.intersection(long.obs_names, sort=False)
    if len(common) < 1000:
        raise ValueError(f"only {len(common)} matched Visium spots")
    short = short[common].copy()
    long = long[common].copy()

    isoform_genes = long.var["ref_gene_name"].astype(str).to_numpy()
    valid_gene = np.isin(isoform_genes, short.var_names.to_numpy())
    valid_gene &= isoform_genes != "nan"
    long = long[:, valid_gene].copy()
    isoform_genes = isoform_genes[valid_gene]
    isoform_structures, recovered_structures = structures_from_annotation(
        args.annotation,
        long.var_names.astype(str).to_numpy(),
        isoform_genes,
        strip_colon_suffix=True,
    )

    obs = pd.DataFrame(index=common.astype(str))
    obs["section"] = long.obs["library_id"].astype(str).to_numpy()
    obs["domain"] = "Visium_Illumina_to_Nanopore"
    obs["spatial_x"] = long.obs["array_col"].to_numpy(dtype=float)
    obs["spatial_y"] = long.obs["array_row"].to_numpy(dtype=float)
    obs["short_umis"] = np.asarray(short.X.sum(axis=1)).ravel()
    obs["long_reads"] = np.asarray(long.X.sum(axis=1)).ravel()

    dataset = PairedDataset(
        x=short.X.tocsr(),
        y=long.X.tocsr(),
        obs=obs,
        genes=short.var_names.astype(str).to_numpy(),
        isoforms=long.var_names.astype(str).to_numpy(),
        isoform_genes=isoform_genes,
        isoform_structures=isoform_structures,
        metadata={
            "category": "spatial",
            "study": "post-myocardial-infarction mouse heart",
            "input": "Visium Illumina gene counts",
            "target": "matched Visium Nanopore isoform counts",
            "pairing": "Visium spot barcode and section",
            "sections": sorted(obs["section"].unique().tolist()),
            "split": "leave-one-section-out",
            "isoform_structures_recovered": recovered_structures,
        },
    )
    save_dataset(dataset, args.output)
    print(
        f"saved {dataset.x.shape[0]} spots, {dataset.x.shape[1]} genes, "
        f"{dataset.y.shape[1]} isoforms to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
