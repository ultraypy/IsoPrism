#!/usr/bin/env python3
"""Harmonize FLAMES/ONT and CRC/PacBio by exact multi-exon splice chains."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from short2long.data import PairedDataset, load_dataset, save_dataset  # noqa: E402


SpliceSignature = tuple[str, str, tuple[tuple[int, int], ...]]


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def attribute(attributes: str, key: str) -> str | None:
    match = re.search(rf"(?:^|;\s*){re.escape(key)}=([^;]+)", attributes)
    return match.group(1) if match else None


def canonical_ensembl(value: str) -> str:
    match = re.search(r"ENS[A-Z]*G\d+", str(value))
    return match.group(0) if match else re.sub(r"\.\d+$", "", str(value))


def parse_isoforms(
    paths: list[Path],
    keep_ids: set[str],
    source: str,
) -> tuple[dict[str, tuple[str, SpliceSignature]], dict[str, str]]:
    """Return matrix isoform -> (Ensembl gene, exact intron-chain signature)."""
    transcripts: dict[str, tuple[str, str, str]] = {}
    exons: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    symbol_to_ensembl_votes: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for path in paths:
        with open_text(path) as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 9:
                    continue
                feature, attrs = fields[2], fields[8]
                if feature == "transcript":
                    tx = attribute(attrs, "ID")
                    if not tx:
                        continue
                    tx = tx.removeprefix("transcript:")
                    if source == "flames":
                        parent = attribute(attrs, "Parent") or ""
                        gene = canonical_ensembl(parent.removeprefix("gene:"))
                    else:
                        gene = canonical_ensembl(attribute(attrs, "associated_gene") or "")
                        symbol = attribute(attrs, "gene_name")
                        if symbol and gene.startswith("ENSG"):
                            symbol_to_ensembl_votes[symbol][gene] += 1
                    if tx in keep_ids:
                        transcripts[tx] = (fields[0], fields[6], gene)
                elif feature == "exon":
                    parent = attribute(attrs, "Parent")
                    if parent:
                        tx = parent.removeprefix("transcript:")
                        if tx in keep_ids:
                            exons[tx].append((int(fields[3]), int(fields[4])))

    parsed: dict[str, tuple[str, SpliceSignature]] = {}
    for tx, (chromosome, strand, gene) in transcripts.items():
        ordered = sorted(exons[tx])
        # Single-exon models cannot be harmonized robustly across discovery tools
        # because small transcript-end shifts create false platform differences.
        if len(ordered) < 2 or not gene.startswith("ENSG"):
            continue
        introns = tuple(
            (ordered[i][1], ordered[i + 1][0]) for i in range(len(ordered) - 1)
        )
        parsed[tx] = (gene, (chromosome, strand, introns))

    symbol_map = {
        symbol: votes.most_common(1)[0][0]
        for symbol, votes in symbol_to_ensembl_votes.items()
    }
    return parsed, symbol_map


def group_columns(
    matrix: sparse.csr_matrix,
    source_labels: np.ndarray,
    target_labels: list[str],
) -> sparse.csr_matrix:
    lookup = {label: index for index, label in enumerate(target_labels)}
    rows: list[int] = []
    columns: list[int] = []
    for row, label in enumerate(source_labels):
        column = lookup.get(str(label))
        if column is not None:
            rows.append(row)
            columns.append(column)
    selector = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(matrix.shape[1], len(target_labels)),
    )
    return (matrix @ selector).tocsr().astype(np.float32)


def aggregate_isoforms(
    matrix: sparse.csr_matrix,
    isoforms: np.ndarray,
    parsed: dict[str, tuple[str, SpliceSignature]],
    keys: list[tuple[str, SpliceSignature]],
) -> sparse.csr_matrix:
    lookup = {key: index for index, key in enumerate(keys)}
    rows: list[int] = []
    columns: list[int] = []
    for row, tx in enumerate(isoforms.astype(str)):
        key = parsed.get(tx)
        if key in lookup:
            rows.append(row)
            columns.append(lookup[key])
    selector = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(matrix.shape[1], len(keys)),
    )
    return (matrix @ selector).tocsr().astype(np.float32)


def signature_id(key: tuple[str, SpliceSignature]) -> str:
    gene, (chromosome, strand, introns) = key
    chain = ";".join(f"{donor}-{acceptor}" for donor, acceptor in introns)
    return f"{gene}|{chromosome}|{strand}|{chain}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flames", type=Path, default=Path("data/processed/flames_cross_dataset")
    )
    parser.add_argument("--crc", type=Path, default=Path("data/processed/crc_paired"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/harmonized_ont_pacbio")
    )
    args = parser.parse_args()

    flames = load_dataset(args.flames)
    crc = load_dataset(args.crc)
    flames_gffs = [
        args.raw_root
        / "flames_cross_dataset_v2"
        / "extracted"
        / "GSM4681740_isoform_annotated.filtered.gff3.gz",
        args.raw_root
        / "flames_cross_dataset_v3"
        / "GSE154870_isoform_annotated.filtered.gff3.gz",
    ]
    crc_gff = (
        args.raw_root
        / "crc_cross_donor"
        / "extracted"
        / "gffcmp.query.all_samples.combined.colored.sorted.gff3"
    )
    flames_v2_parsed, _ = parse_isoforms(
        [flames_gffs[0]], set(flames.isoforms.astype(str)), "flames"
    )
    flames_v3_parsed, _ = parse_isoforms(
        [flames_gffs[1]], set(flames.isoforms.astype(str)), "flames"
    )
    crc_parsed, symbol_map = parse_isoforms(
        [crc_gff], set(crc.isoforms.astype(str)), "crc"
    )

    crc_keys: defaultdict[tuple[str, SpliceSignature], list[str]] = defaultdict(list)
    for tx, key in crc_parsed.items():
        crc_keys[key].append(tx)
    # The two FLAMES releases reused some transcript identifiers for slightly
    # different structures.  Intersect biological splice-chain keys rather than
    # identifiers so that both ONT datasets share an unambiguous target catalog.
    common_keys = sorted(
        set(flames_v2_parsed.values())
        & set(flames_v3_parsed.values())
        & set(crc_keys),
        key=signature_id,
    )

    flames_gene_labels = np.asarray(
        [canonical_ensembl(gene) for gene in flames.genes.astype(str)]
    )
    crc_gene_labels = np.asarray(
        [symbol_map.get(str(gene), "") for gene in crc.genes.astype(str)]
    )
    shared_genes = sorted(
        (set(flames_gene_labels) & set(crc_gene_labels)) - {""}
    )
    shared_gene_set = set(shared_genes)
    common_keys = [key for key in common_keys if key[0] in shared_gene_set]
    if len(common_keys) < 500:
        raise ValueError(f"only {len(common_keys)} shared multi-exon splice chains")

    v2_rows = np.flatnonzero(
        flames.obs["dataset"].astype(str).to_numpy() == "GSE154869_10x_v2"
    )
    v3_rows = np.flatnonzero(
        flames.obs["dataset"].astype(str).to_numpy() == "GSE154870_10x_v3"
    )
    flames_order = np.concatenate([v2_rows, v3_rows])
    flames_x = group_columns(flames.x[flames_order], flames_gene_labels, shared_genes)
    crc_x = group_columns(crc.x, crc_gene_labels, shared_genes)
    flames_y = sparse.vstack(
        [
            aggregate_isoforms(
                flames.y[v2_rows], flames.isoforms, flames_v2_parsed, common_keys
            ),
            aggregate_isoforms(
                flames.y[v3_rows], flames.isoforms, flames_v3_parsed, common_keys
            ),
        ]
    ).tocsr()
    crc_y = aggregate_isoforms(crc.y, crc.isoforms, crc_parsed, common_keys)

    # Keep targets observed on both platforms; platform-specific zero columns do
    # not measure transfer and make the allocation metrics artificially easy.
    observed = (np.asarray(flames_y.sum(axis=0)).ravel() > 0) & (
        np.asarray(crc_y.sum(axis=0)).ravel() > 0
    )
    common_keys = [key for key, keep in zip(common_keys, observed) if keep]
    flames_y = flames_y[:, observed]
    crc_y = crc_y[:, observed]

    flames_obs = flames.obs.iloc[flames_order].copy()
    flames_obs.index = [f"ONT:{cell}" for cell in flames_obs.index.astype(str)]
    flames_obs["domain"] = "FLAMES_ONT"
    flames_obs["platform"] = "Oxford_Nanopore"
    flames_obs["study"] = "GSE154869_GSE154870"
    crc_obs = crc.obs.copy()
    crc_obs.index = [f"PacBio:{cell}" for cell in crc_obs.index.astype(str)]
    crc_obs["domain"] = "CRC_PacBio"
    crc_obs["platform"] = "PacBio"
    crc_obs["study"] = "CRC_long_read_atlas"

    dataset = PairedDataset(
        x=sparse.vstack([flames_x, crc_x]).tocsr(),
        y=sparse.vstack([flames_y, crc_y]).tocsr(),
        obs=pd.concat([flames_obs, crc_obs], axis=0),
        genes=np.asarray(shared_genes),
        isoforms=np.asarray([signature_id(key) for key in common_keys]),
        isoform_genes=np.asarray([key[0] for key in common_keys]),
        metadata={
            "categories": ["cross_dataset", "cross_platform"],
            "harmonization": "exact chromosome/strand/intron-chain match; single-exon models excluded",
            "genome_build": "GRCh38",
            "domains": {
                "FLAMES_ONT": {
                    "study": "GSE154869 + GSE154870",
                    "platform": "Oxford Nanopore",
                    "cells": int(len(flames_obs)),
                },
                "CRC_PacBio": {
                    "study": "colorectal cancer long-read single-cell atlas",
                    "platform": "PacBio",
                    "cells": int(len(crc_obs)),
                },
            },
            "cross_dataset_design": "train CRC_PacBio; evaluate all FLAMES_ONT cells",
            "cross_platform_design": "train FLAMES_ONT; evaluate all CRC_PacBio cells",
            "n_shared_input_genes": len(shared_genes),
            "n_shared_splice_chains": len(common_keys),
        },
    )
    save_dataset(dataset, args.output)
    print(json.dumps(dataset.metadata, ensure_ascii=False, indent=2))
    print(
        f"saved {dataset.x.shape[0]} cells, {dataset.x.shape[1]} genes, "
        f"{dataset.y.shape[1]} shared isoforms to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
