#!/usr/bin/env python3
"""Prepare the two FLAMES scmixology datasets for cross-dataset evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from short2long.data import PairedDataset, save_dataset  # noqa: E402


def load_wide(path: Path, index_columns: int) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray | None]:
    frame = pd.read_csv(path)
    ids = frame.iloc[:, 0].astype(str).to_numpy()
    parents = frame.iloc[:, 1].astype(str).to_numpy() if index_columns == 2 else None
    cells = frame.columns[index_columns:].astype(str).to_numpy()
    values = sparse.csr_matrix(frame.iloc[:, index_columns:].to_numpy(dtype=np.float32).T)
    return values, cells, ids, parents


def load_barcode_map(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    return dict(zip(frame["cell_name"].astype(str), frame["barcode_sequence"].astype(str)))


def prepare_one(dataset_id: str, gene_path: Path, transcript_path: Path, map_path: Path) -> PairedDataset:
    x, x_cells, genes, _ = load_wide(gene_path, 1)
    y, y_cells, isoforms, isoform_genes = load_wide(transcript_path, 2)
    barcode_map = load_barcode_map(map_path)
    x_barcodes = np.asarray([barcode_map.get(cell, cell) for cell in x_cells])

    # Illumina has more barcodes; retain only cells with long-read truth.
    x_lookup = {cell: i for i, cell in enumerate(x_barcodes)}
    y_lookup = {cell: i for i, cell in enumerate(y_cells)}
    common = [cell for cell in y_cells if cell in x_lookup]
    if len(common) < 100:
        raise ValueError(f"only {len(common)} paired barcodes in {dataset_id}")
    x = x[[x_lookup[cell] for cell in common]].tocsr()
    y = y[[y_lookup[cell] for cell in common]].tocsr()

    keep_gene = np.asarray(x.sum(axis=0)).ravel() >= 5
    keep_iso = (np.asarray(y.sum(axis=0)).ravel() >= 10) & (np.asarray((y > 0).sum(axis=0)).ravel() >= 3)
    x = x[:, keep_gene]
    genes = genes[keep_gene]
    y = y[:, keep_iso]
    isoforms = isoforms[keep_iso]
    isoform_genes = isoform_genes[keep_iso]

    obs = pd.DataFrame(
        {"dataset": dataset_id, "domain": dataset_id, "barcode": common},
        index=[f"{dataset_id}:{cell}" for cell in common],
    )
    return PairedDataset(
        x=x,
        y=y,
        obs=obs,
        genes=genes,
        isoforms=isoforms,
        isoform_genes=isoform_genes,
        metadata={
            "category": "cross_dataset",
            "input": "matched short-read gene count matrix",
            "target": "Nanopore isoform count matrix",
            "pairing": "10x cell barcode",
            "dataset_id": dataset_id,
        },
    )


def align_and_combine(a: PairedDataset, b: PairedDataset) -> PairedDataset:
    common_genes = sorted(set(a.genes) & set(b.genes))
    common_isoforms = sorted(set(a.isoforms) & set(b.isoforms))
    if len(common_genes) < 500 or len(common_isoforms) < 500:
        raise ValueError("insufficient cross-dataset feature overlap")
    ai_g = {v: i for i, v in enumerate(a.genes)}
    bi_g = {v: i for i, v in enumerate(b.genes)}
    ai_t = {v: i for i, v in enumerate(a.isoforms)}
    bi_t = {v: i for i, v in enumerate(b.isoforms)}
    a_tx_gene = dict(zip(a.isoforms, a.isoform_genes))
    b_tx_gene = dict(zip(b.isoforms, b.isoform_genes))
    isoform_genes = np.asarray([a_tx_gene.get(t, b_tx_gene[t]) for t in common_isoforms])
    combined = PairedDataset(
        x=sparse.vstack(
            [a.x[:, [ai_g[v] for v in common_genes]], b.x[:, [bi_g[v] for v in common_genes]]]
        ).tocsr(),
        y=sparse.vstack(
            [a.y[:, [ai_t[v] for v in common_isoforms]], b.y[:, [bi_t[v] for v in common_isoforms]]]
        ).tocsr(),
        obs=pd.concat([a.obs, b.obs], axis=0),
        genes=np.asarray(common_genes),
        isoforms=np.asarray(common_isoforms),
        isoform_genes=isoform_genes,
        metadata={
            "category": "cross_dataset",
            "train_domain": "GSE154869_10x_v2",
            "test_domain": "GSE154870_10x_v3",
            "pairing": "within-dataset 10x cell barcode",
            "n_shared_genes": len(common_genes),
            "n_shared_isoforms": len(common_isoforms),
        },
    )
    combined.validate()
    return combined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/flames_cross_dataset"))
    args = parser.parse_args()
    v2 = args.raw_root / "flames_cross_dataset_v2" / "extracted"
    v3e = args.raw_root / "flames_cross_dataset_v3" / "extracted"
    v3 = args.raw_root / "flames_cross_dataset_v3"
    a = prepare_one(
        "GSE154869_10x_v2",
        v2 / "GSM4681740_gene_count_Lib10.csv.gz",
        v2 / "GSM4681740_transcript_count.csv.gz",
        v2 / "GSM4681740_Lib10.csv.gz",
    )
    b = prepare_one(
        "GSE154870_10x_v3",
        v3e / "GSM4681741_gene_count_Lib10.csv.gz",
        v3 / "GSE154870_transcript_count.csv.gz",
        v3e / "GSM4681741_Lib10.csv.gz",
    )
    combined = align_and_combine(a, b)
    save_dataset(combined, args.output)
    print(
        f"saved {combined.x.shape[0]} cells, {combined.x.shape[1]} genes, "
        f"{combined.y.shape[1]} isoforms to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
