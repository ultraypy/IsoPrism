from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class PairedDataset:
    x: sparse.csr_matrix
    y: sparse.csr_matrix
    obs: pd.DataFrame
    genes: np.ndarray
    isoforms: np.ndarray
    isoform_genes: np.ndarray
    metadata: dict
    isoform_structures: np.ndarray | None = None

    def validate(self) -> None:
        if not sparse.isspmatrix_csr(self.x):
            self.x = self.x.tocsr()
        if not sparse.isspmatrix_csr(self.y):
            self.y = self.y.tocsr()
        if self.x.shape != (len(self.obs), len(self.genes)):
            raise ValueError(f"x shape {self.x.shape} does not match obs/genes")
        if self.y.shape != (len(self.obs), len(self.isoforms)):
            raise ValueError(f"y shape {self.y.shape} does not match obs/isoforms")
        if len(self.isoforms) != len(self.isoform_genes):
            raise ValueError("every isoform requires one parent gene")
        if self.isoform_structures is not None and len(self.isoform_structures) != len(
            self.isoforms
        ):
            raise ValueError("every isoform requires one structural identifier")
        if self.obs.index.has_duplicates:
            raise ValueError("cell ids must be unique")
        if len(set(map(str, self.genes))) != len(self.genes):
            raise ValueError("gene ids must be unique")
        if len(set(map(str, self.isoforms))) != len(self.isoforms):
            raise ValueError("isoform ids must be unique")


def save_dataset(dataset: PairedDataset, directory: Path) -> None:
    dataset.validate()
    directory.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(directory / "x_gene_counts.npz", dataset.x, compressed=True)
    sparse.save_npz(directory / "y_isoform_counts.npz", dataset.y, compressed=True)
    dataset.obs.to_csv(directory / "obs.csv", index=True, index_label="cell_id")
    pd.DataFrame({"gene_id": dataset.genes}).to_csv(directory / "genes.csv", index=False)
    isoform_table = {
        "isoform_id": dataset.isoforms,
        "gene_id": dataset.isoform_genes,
    }
    if dataset.isoform_structures is not None:
        isoform_table["structure_id"] = dataset.isoform_structures
    pd.DataFrame(isoform_table).to_csv(directory / "isoforms.csv", index=False)
    (directory / "dataset.json").write_text(
        json.dumps(dataset.metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_dataset(directory: Path) -> PairedDataset:
    obs = pd.read_csv(directory / "obs.csv", index_col="cell_id")
    genes = pd.read_csv(directory / "genes.csv")["gene_id"].astype(str).to_numpy()
    iso = pd.read_csv(directory / "isoforms.csv", dtype=str)
    metadata = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    dataset = PairedDataset(
        x=sparse.load_npz(directory / "x_gene_counts.npz").tocsr(),
        y=sparse.load_npz(directory / "y_isoform_counts.npz").tocsr(),
        obs=obs,
        genes=genes,
        isoforms=iso["isoform_id"].to_numpy(),
        isoform_genes=iso["gene_id"].to_numpy(),
        metadata=metadata,
        isoform_structures=(
            iso["structure_id"].to_numpy()
            if "structure_id" in iso.columns
            else None
        ),
    )
    dataset.validate()
    return dataset


def normalize_log_cpm(matrix: sparse.csr_matrix, scale: float = 1e4) -> np.ndarray:
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    totals[totals <= 0] = 1.0
    normalized = matrix.multiply((scale / totals)[:, None]).toarray().astype(np.float32)
    return np.log1p(normalized)


def choose_input_genes(matrix: sparse.csr_matrix, max_genes: int) -> np.ndarray:
    """Select variable genes without densifying the full cell-by-gene matrix."""
    if matrix.shape[1] <= max_genes:
        return np.arange(matrix.shape[1], dtype=np.int64)
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    mean_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    variance = np.maximum(mean_sq - mean**2, 0)
    dispersion = variance / np.maximum(mean, 1e-3)
    return np.sort(np.argpartition(dispersion, -max_genes)[-max_genes:]).astype(np.int64)
