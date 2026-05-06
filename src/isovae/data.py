from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import set_seed

Array = np.ndarray


def _as_array(x: Any) -> Array:
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def strip_barcode_suffix(index: pd.Index) -> pd.Index:
    """Normalize 10x-style cell barcodes by removing a trailing ``-1`` suffix."""
    return index.astype(str).str.replace(r"-1$", "", regex=True)


def align_paired_cells(
    adata_gene: ad.AnnData,
    adata_iso: ad.AnnData,
    strict: bool = True,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """Align paired short-read and long-read AnnData objects by cell barcode."""
    gene_barcodes = strip_barcode_suffix(adata_gene.obs_names)
    iso_barcodes = strip_barcode_suffix(adata_iso.obs_names)

    gene_pos = {barcode: i for i, barcode in enumerate(gene_barcodes)}
    iso_pos = {barcode: i for i, barcode in enumerate(iso_barcodes)}
    common = sorted(set(gene_pos) & set(iso_pos))
    if strict and not common:
        raise ValueError("No common cells found after barcode normalization.")

    gene_idx = [gene_pos[barcode] for barcode in common]
    iso_idx = [iso_pos[barcode] for barcode in common]
    gene_aligned = adata_gene[gene_idx, :].copy()
    iso_aligned = adata_iso[iso_idx, :].copy()
    normalized_index = pd.Index(common, name="barcode")
    gene_aligned.obs["barcode_raw"] = gene_aligned.obs_names.astype(str)
    iso_aligned.obs["barcode_raw"] = iso_aligned.obs_names.astype(str)
    gene_aligned.obs_names = normalized_index
    iso_aligned.obs_names = normalized_index
    return gene_aligned, iso_aligned


def make_unique_gene_symbol_view(adata_gene: ad.AnnData) -> ad.AnnData:
    """Return a copy with unique gene-symbol columns."""
    if "gene_symbol" in adata_gene.var.columns:
        symbols = adata_gene.var["gene_symbol"].astype(str).values
    else:
        symbols = adata_gene.var_names.astype(str).values
        adata_gene = adata_gene.copy()
        adata_gene.var["gene_symbol"] = symbols
    keep = ~pd.Index(symbols).duplicated(keep="first")
    return adata_gene[:, keep].copy()


def _col_var(x: Any) -> Array:
    if sp.issparse(x):
        mean = np.asarray(x.mean(axis=0)).ravel()
        sq_mean = np.asarray(x.power(2).mean(axis=0)).ravel()
        return sq_mean - mean**2
    return np.asarray(x).var(axis=0)


def normalize_gene_counts(x: Any) -> Array:
    """Library-size normalize selected short-read gene counts and apply log1p."""
    if sp.issparse(x):
        lib = np.asarray(x.sum(axis=1)).ravel().astype(np.float32)
        scale = (1e4 / np.maximum(lib, 1e-6)).astype(np.float32)
        return x.multiply(scale[:, None]).log1p().toarray().astype(np.float32)

    x = np.asarray(x, dtype=np.float32)
    lib = x.sum(axis=1, keepdims=True)
    return np.log1p((x / np.maximum(lib, 1e-6)) * 1e4).astype(np.float32)


def counts_to_gene_usage(
    y_counts: Any,
    isoform_gene: Sequence[str],
) -> Tuple[Array, List[Array], Array]:
    """Convert transcript counts to within-gene isoform-usage proportions."""
    y_counts = _as_array(y_counts).astype(np.float32)
    gene_to_idx: Dict[str, List[int]] = {}
    for j, g in enumerate(np.asarray(isoform_gene).astype(str)):
        gene_to_idx.setdefault(g, []).append(j)

    genes = np.array(list(gene_to_idx.keys()), dtype=object)
    groups = [np.array(v, dtype=np.int64) for v in gene_to_idx.values()]
    y_usage = np.zeros_like(y_counts, dtype=np.float32)
    for idx in groups:
        denom = y_counts[:, idx].sum(axis=1, keepdims=True)
        rows = np.where(denom[:, 0] > 0)[0]
        if rows.size:
            y_usage[np.ix_(rows, idx)] = y_counts[np.ix_(rows, idx)] / denom[rows]
    return y_usage, groups, genes


@dataclass
class IsoVAEPreprocessor:
    """Feature names and scaler needed for short-read-only prediction."""

    genes: List[str]
    isoforms: List[str]
    isoform_gene: List[str]
    gene_groups: List[Array]
    scaler: StandardScaler

    def transform_gene_adata(self, adata_gene: ad.AnnData) -> Tuple[Array, int]:
        """Extract and scale the model input genes from a short-read AnnData object."""
        adata_gene = make_unique_gene_symbol_view(adata_gene)
        symbols = adata_gene.var["gene_symbol"].astype(str).values
        symbol_to_pos = {g: i for i, g in enumerate(symbols)}
        cols = []
        n_found = 0
        for gene in self.genes:
            if gene in symbol_to_pos:
                cols.append(adata_gene.X[:, symbol_to_pos[gene]])
                n_found += 1
            else:
                cols.append(sp.csr_matrix((adata_gene.n_obs, 1), dtype=np.float32))
        x_raw = sp.hstack(cols, format="csr") if sp.issparse(cols[0]) else np.column_stack(cols)
        x = normalize_gene_counts(x_raw)
        return self.scaler.transform(x).astype(np.float32), n_found

    def extract_iso_counts(self, adata_iso: ad.AnnData) -> Tuple[Array, int]:
        """Extract model isoform-count features from a long-read AnnData object."""
        transcript = (
            adata_iso.var["transcript_id"].astype(str).values
            if "transcript_id" in adata_iso.var.columns
            else adata_iso.var_names.astype(str).values
        )
        tx_to_pos = {t: i for i, t in enumerate(transcript)}
        cols = []
        n_found = 0
        for tx in self.isoforms:
            if tx in tx_to_pos:
                cols.append(adata_iso.X[:, tx_to_pos[tx]])
                n_found += 1
            else:
                cols.append(sp.csr_matrix((adata_iso.n_obs, 1), dtype=np.float32))
        y = sp.hstack(cols, format="csr") if sp.issparse(cols[0]) else np.column_stack(cols)
        return _as_array(y).astype(np.float32), n_found


def prepare_paired_data(
    adata_gene: ad.AnnData,
    adata_iso: ad.AnnData,
    gene_hvg: int = 1200,
    min_iso_cells: int = 10,
    test_size: float = 0.20,
    val_size_within_train: float = 0.20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Prepare paired data for training/evaluation.

    Feature selection and scaler fitting are performed using training cells only.
    """
    set_seed(seed)
    adata_gene, adata_iso = align_paired_cells(adata_gene, adata_iso)
    adata_gene = make_unique_gene_symbol_view(adata_gene)

    all_idx = np.arange(adata_gene.n_obs)
    train_idx, test_idx = train_test_split(all_idx, test_size=test_size, random_state=seed)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_size_within_train, random_state=seed
    )

    gene_symbols = adata_gene.var["gene_symbol"].astype(str).values
    iso_gene_all = adata_iso.var["gene_id"].astype(str).values
    common_genes = np.intersect1d(np.unique(gene_symbols), np.unique(iso_gene_all))
    if common_genes.size == 0:
        raise ValueError("No overlap between short-read gene_symbol and LR isoform gene_id.")

    common_mask = np.isin(gene_symbols, common_genes)
    adata_gene_common = adata_gene[:, common_mask]
    var = _col_var(adata_gene_common.X[train_idx, :])
    top_idx = np.argsort(var)[::-1][: min(gene_hvg, adata_gene_common.n_vars)]
    selected_genes = adata_gene_common.var["gene_symbol"].astype(str).values[top_idx]

    iso_mask = np.isin(iso_gene_all, selected_genes)
    adata_iso_sel = adata_iso[:, iso_mask].copy()
    iso_nonzero_cells = np.asarray((adata_iso_sel.X[train_idx, :] > 0).sum(axis=0)).ravel()
    adata_iso_sel = adata_iso_sel[:, iso_nonzero_cells >= min_iso_cells].copy()

    iso_gene = adata_iso_sel.var["gene_id"].astype(str).values
    gene_to_idx: Dict[str, List[int]] = {}
    for j, g in enumerate(iso_gene):
        gene_to_idx.setdefault(g, []).append(j)
    genes_keep = [g for g, idx in gene_to_idx.items() if len(idx) >= 2]
    if not genes_keep:
        raise ValueError("No genes with >=2 isoforms after filtering.")

    iso_keep_idx = np.concatenate([gene_to_idx[g] for g in genes_keep]).astype(np.int64)
    adata_iso_sel = adata_iso_sel[:, iso_keep_idx].copy()
    isoform_gene = adata_iso_sel.var["gene_id"].astype(str).values
    isoforms = (
        adata_iso_sel.var["transcript_id"].astype(str).values
        if "transcript_id" in adata_iso_sel.var.columns
        else adata_iso_sel.var_names.astype(str).values
    )
    y_counts = _as_array(adata_iso_sel.X).astype(np.float32)
    y_usage, gene_groups, genes = counts_to_gene_usage(y_counts, isoform_gene)
    genes = genes.astype(str)

    # Build short-read inputs in retained gene-group order.
    # Manual extraction before scaler is fitted.
    adata_gene_u = make_unique_gene_symbol_view(adata_gene)
    symbols = adata_gene_u.var["gene_symbol"].astype(str).values
    symbol_to_pos = {g: i for i, g in enumerate(symbols)}
    cols = []
    n_gene_found = 0
    for g in genes:
        if g in symbol_to_pos:
            cols.append(adata_gene_u.X[:, symbol_to_pos[g]])
            n_gene_found += 1
        else:
            cols.append(sp.csr_matrix((adata_gene_u.n_obs, 1), dtype=np.float32))
    x_raw = sp.hstack(cols, format="csr") if sp.issparse(cols[0]) else np.column_stack(cols)
    x_norm = normalize_gene_counts(x_raw)
    scaler = StandardScaler().fit(x_norm[train_idx])
    x_all = scaler.transform(x_norm).astype(np.float32)

    preprocessor = IsoVAEPreprocessor(
        genes=list(map(str, genes)),
        isoforms=list(map(str, isoforms)),
        isoform_gene=list(map(str, isoform_gene)),
        gene_groups=gene_groups,
        scaler=scaler,
    )
    return {
        "x_train": x_all[train_idx],
        "x_val": x_all[val_idx],
        "x_test": x_all[test_idx],
        "y_train_usage": y_usage[train_idx],
        "y_val_usage": y_usage[val_idx],
        "y_test_usage": y_usage[test_idx],
        "y_train_counts": y_counts[train_idx],
        "y_val_counts": y_counts[val_idx],
        "y_test_counts": y_counts[test_idx],
        "train_obs_names": adata_gene.obs_names[train_idx].astype(str).tolist(),
        "val_obs_names": adata_gene.obs_names[val_idx].astype(str).tolist(),
        "test_obs_names": adata_gene.obs_names[test_idx].astype(str).tolist(),
        "preprocessor": preprocessor,
        "gene_groups": gene_groups,
        "genes_final": genes.astype(str),
        "isoforms_final": np.asarray(isoforms).astype(str),
        "isoform_gene": np.asarray(isoform_gene).astype(str),
        "meta": {
            "n_cells": int(adata_gene.n_obs),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "n_gene_groups": int(len(gene_groups)),
            "n_isoforms_output": int(len(isoforms)),
            "n_gene_features_found": int(n_gene_found),
        },
    }
