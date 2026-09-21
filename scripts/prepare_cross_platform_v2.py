"""Recover shared splice chains before intersecting study-specific identifiers.

This creates a separate data version; it never modifies the original benchmark.
No target-domain count is consulted when constructing the shared catalogue.
The catalogue still uses both studies' released annotations (closed catalogue,
not de-novo discovery), and proportions are conditional on retained chains.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from short2long.data import PairedDataset, load_dataset, save_dataset
from prepare_flames import load_wide, load_barcode_map
from prepare_cross_study_platform import (
    parse_isoforms, canonical_ensembl, signature_id, group_columns, aggregate_isoforms,
)


def load_unfiltered(dataset_id, gene_path, transcript_path, map_path):
    x, x_cells, genes, _ = load_wide(gene_path, 1)
    y, y_cells, isoforms, parents = load_wide(transcript_path, 2)
    mapping = load_barcode_map(map_path)
    barcodes = np.asarray([mapping.get(c, c) for c in x_cells])
    if len(set(barcodes)) != len(barcodes) or len(set(y_cells)) != len(y_cells):
        raise ValueError("ambiguous or duplicate cell barcode")
    xi = {c: i for i, c in enumerate(barcodes)}
    yi = {c: i for i, c in enumerate(y_cells)}
    common = [c for c in y_cells if c in xi]
    obs = pd.DataFrame({"dataset": dataset_id, "barcode": common},
                       index=[f"{dataset_id}:{c}" for c in common])
    return PairedDataset(x=x[[xi[c] for c in common]].tocsr(),
                         y=y[[yi[c] for c in common]].tocsr(), obs=obs,
                         genes=genes, isoforms=isoforms, isoform_genes=parents,
                         metadata={"pairing": "matched Illumina and ONT barcode"})


def structural_catalogue(parsed_by_dataset, shared_genes):
    """Only annotation dictionaries, never expression matrices, enter here.

    Require presence in EACH quantification catalogue so that a missing assay
    target is not silently converted to a biological zero. Transcript IDs are
    local to a release and are not used as cross-release join keys.
    """
    common = set.intersection(*(set(p.values()) for p in parsed_by_dataset))
    return sorted((k for k in common if k[0] in shared_genes), key=signature_id)


def extract_author_pair(root, number):
    """Extract only the two explicitly named, non-executable matrix resources."""
    archive = root / f"PromethION_scmixology{number}.zip"
    out = root / "extracted" / f"rep{number}"
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in ["transcript_count.csv.gz", "isoform_annotated.filtered.gff3"]:
            member = f"PromethION_5cl_rep{number}/FLTSA_output/{name}"
            data = z.read(member)  # verifies the ZIP member CRC
            target = out / name
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError(f"refusing to replace different extracted data: {target}")
            else:
                target.write_bytes(data)
    return out


def build(raw_root, crc_path, author_release=None):
    v2 = raw_root / "flames_cross_dataset_v2" / "extracted"
    v3 = raw_root / "flames_cross_dataset_v3"
    a_count, a_annotation = v2 / "GSM4681740_transcript_count.csv.gz", v2 / "GSM4681740_isoform_annotated.filtered.gff3.gz"
    b_count, b_annotation = v3 / "GSE154870_transcript_count.csv.gz", v3 / "GSE154870_isoform_annotated.filtered.gff3.gz"
    if author_release is not None:
        aa, bb = extract_author_pair(author_release, 1), extract_author_pair(author_release, 2)
        a_count, a_annotation = aa / "transcript_count.csv.gz", aa / "isoform_annotated.filtered.gff3"
        b_count, b_annotation = bb / "transcript_count.csv.gz", bb / "isoform_annotated.filtered.gff3"
    a = load_unfiltered("GSE154869_10x_v2", v2 / "GSM4681740_gene_count_Lib10.csv.gz",
                        a_count, v2 / "GSM4681740_Lib10.csv.gz")
    b = load_unfiltered("GSE154870_10x_v3", v3 / "extracted/GSM4681741_gene_count_Lib10.csv.gz",
                        b_count, v3 / "extracted/GSM4681741_Lib10.csv.gz")
    c = load_dataset(crc_path)
    pa, _ = parse_isoforms([a_annotation], set(a.isoforms), "flames")
    pb, _ = parse_isoforms([b_annotation], set(b.isoforms), "flames")
    pc, symbols = parse_isoforms([raw_root / "crc_cross_donor/extracted/gffcmp.query.all_samples.combined.colored.sorted.gff3"], set(c.isoforms), "crc")
    gene_labels = [np.asarray([canonical_ensembl(g) for g in d.genes]) for d in (a, b)]
    gene_labels.append(np.asarray([symbols.get(g, "") for g in c.genes]))
    genes = sorted(set.intersection(*(set(g) for g in gene_labels)) - {""})
    parsed = [pa, pb, pc]
    keys = structural_catalogue(parsed, set(genes))
    xs, ys, observations = [], [], []
    mass = []
    for d, p, gl, prefix, domain, platform in zip(
        [a, b, c], parsed, gene_labels, ["ONT", "ONT", "PacBio"],
        ["FLAMES_ONT", "FLAMES_ONT", "CRC_PacBio"],
        ["Oxford_Nanopore", "Oxford_Nanopore", "PacBio"]):
        xs.append(group_columns(d.x, gl, genes))
        yy = aggregate_isoforms(d.y, d.isoforms, p, keys)
        ys.append(yy)
        obs = d.obs.copy()
        obs.index = [f"{prefix}:{i}" for i in obs.index]
        obs["domain"], obs["platform"] = domain, platform
        observations.append(obs)
        original_mass, retained_mass = float(d.y.sum(dtype=np.float64)), float(yy.sum(dtype=np.float64))
        if retained_mass > original_mass * 1.00001:
            raise ValueError("harmonization duplicated counts")
        mass.append({"domain": domain, "cells": len(d.obs), "raw_isoforms": len(d.isoforms),
                     "parsed_multi_exon_isoforms": len(p), "original_count_sum": original_mass,
                     "retained_count_sum": retained_mass,
                     "retained_count_fraction": retained_mass / max(original_mass, 1)})
    same_ids = set(pa) & set(pb)
    meta = {"version": "structure_first_v2", "genome_build": "GRCh38",
            "input": "independently measured matched Illumina gene expression; CRC is inverse-log normalized expression, not raw UMI",
            "target": "original measured long-read counts, aggregated by exact intron chain",
            "harmonization": "common GRCh38 chromosome/strand/full intron chains in all three released quantification catalogues; single exon excluded",
            "target_count_based_catalogue_filter": False,
            "uses_external_annotation_catalogue": True,
            "proportion_estimand": "conditional within-gene allocation among retained common splice chains, NOT all full-length isoforms",
            "cross_platform_design": "train FLAMES_ONT; evaluate CRC_PacBio; study, tissue and protocol are also confounded",
            "n_shared_input_genes": len(genes), "n_shared_splice_chains": len(keys),
            "flames_same_ids_different_structures": sum(pa[i] != pb[i] for i in same_ids),
            "mass_audit": mass,
            "not_model_results": True}
    if author_release is not None:
        meta.update(version="author_matched_release_v3",
                    author_source="https://github.com/LuyiTian/FLTseq_data/tree/720852b72b7b310c41349af28e9a84c7228bb246/data",
                    source_matrix_note="author reprocessed counts paired by barcode to the original GEO Illumina matrices; source cell sets differ from v1/v2; all compatible provided cells retained",
                    long_read_sources=[str(a_count), str(b_count), str(crc_path)])
    dataset = PairedDataset(x=sparse.vstack(xs).tocsr(), y=sparse.vstack(ys).tocsr(),
                           obs=pd.concat(observations), genes=np.asarray(genes),
                           isoforms=np.asarray([signature_id(k) for k in keys]),
                           isoform_genes=np.asarray([k[0] for k in keys]),
                           isoform_structures=np.asarray([signature_id(k) for k in keys]), metadata=meta)
    dataset.validate()
    return dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--crc", type=Path, default=Path("data/processed/crc_paired"))
    ap.add_argument("--output", type=Path, default=Path("data/processed/harmonized_ont_pacbio_v2"))
    ap.add_argument("--author-release", type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    d = build(args.raw_root, args.crc, args.author_release)
    save_dataset(d, args.output)
    print(json.dumps(d.metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
