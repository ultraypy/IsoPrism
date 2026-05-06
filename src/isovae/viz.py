from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_gene_usage_boxplot(
    usage_long: pd.DataFrame,
    gene: str,
    groupby: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot isoform-usage distributions for one gene.

    ``usage_long`` should contain at least columns: ``gene``, ``isoform``,
    ``usage`` and optionally a grouping column such as stage or cell type.
    """
    data = usage_long.loc[usage_long["gene"].astype(str) == str(gene)].copy()
    if data.empty:
        raise ValueError(f"No rows found for gene={gene!r}.")
    if ax is None:
        _, ax = plt.subplots(figsize=(max(5, data["isoform"].nunique() * 1.2), 4))
    if groupby and groupby in data.columns:
        sns.boxplot(data=data, x=groupby, y="usage", hue="isoform", ax=ax, fliersize=0.5)
        ax.tick_params(axis="x", rotation=45)
    else:
        sns.violinplot(data=data, x="isoform", y="usage", ax=ax, cut=0, inner="box")
        ax.tick_params(axis="x", rotation=45)
    ax.set_title(f"{gene} isoform usage")
    ax.set_xlabel(groupby if groupby else "Isoform")
    ax.set_ylabel("Isoform usage")
    return ax


def plot_usage_heatmap(
    usage: pd.DataFrame,
    isoforms: Optional[Sequence[str]] = None,
    max_cells: int = 200,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a compact heatmap of cell-by-isoform usage values."""
    mat = usage.loc[:, list(isoforms)] if isoforms is not None else usage
    if mat.shape[0] > max_cells:
        mat = mat.iloc[:max_cells]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(mat, cmap="viridis", xticklabels=False, yticklabels=False, ax=ax)
    ax.set_xlabel("Isoforms")
    ax.set_ylabel("Cells")
    return ax


def savefig(path: str | Path, dpi: int = 300) -> None:
    """Save current matplotlib figure with tight layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
