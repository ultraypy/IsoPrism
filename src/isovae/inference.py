from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import (
    IsoVAEPreprocessor,
    align_paired_cells,
    counts_to_gene_usage,
    make_unique_gene_symbol_view,
    normalize_gene_counts,
)
from .model import (
    IsoVAEModel,
    iso_encoder_input_from_counts,
    logits_to_usage,
    make_config_from_checkpoint,
)
from .utils import select_device

Array = np.ndarray


@dataclass
class IsoVAEArtifact:
    """Loaded model plus feature metadata.

    A fitted ``preprocessor`` is required for short-read-only prediction.
    Older checkpoints contain feature names but not the scaler; in that case,
    pass a preprocessor reconstructed from the original training data.
    """

    model: IsoVAEModel
    genes: List[str]
    isoforms: List[str]
    isoform_gene: List[str]
    gene_groups: List[Array]
    preprocessor: Optional[IsoVAEPreprocessor] = None
    device: str = "cpu"

    def to(self, device: str) -> "IsoVAEArtifact":
        self.device = device
        self.model.to(device)
        self.model.eval()
        return self


def load_artifact(
    checkpoint: str | Path,
    preprocessor: Optional[IsoVAEPreprocessor] = None,
    device: Optional[str] = None,
) -> IsoVAEArtifact:
    """Load an IsoVAE checkpoint saved by the training scripts."""
    device = device or select_device(prefer_cuda=True)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    genes = list(map(str, ckpt["genes_final"]))
    isoforms = list(map(str, ckpt["isoforms_final"]))
    isoform_gene = list(map(str, ckpt["isoform_gene"]))
    groups = [np.asarray(g, dtype=np.int64) for g in ckpt["gene_groups"]]
    config = make_config_from_checkpoint(ckpt.get("model_config", {}), ckpt.get("model_state"))
    model = IsoVAEModel(
        n_gene_inputs=len(genes),
        n_isoforms=len(isoforms),
        n_gene_groups=len(groups),
        config=config,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    if preprocessor is not None:
        # Ensure model and preprocessor features agree.
        if list(preprocessor.genes) != genes:
            raise ValueError("Preprocessor genes do not match checkpoint genes.")
        if list(preprocessor.isoforms) != isoforms:
            raise ValueError("Preprocessor isoforms do not match checkpoint isoforms.")

    return IsoVAEArtifact(
        model=model,
        genes=genes,
        isoforms=isoforms,
        isoform_gene=isoform_gene,
        gene_groups=groups,
        preprocessor=preprocessor,
        device=device,
    )


def reconstruct_preprocessor_from_training_data(
    checkpoint: str | Path,
    adata_gene_train: ad.AnnData,
    adata_iso_train: Optional[ad.AnnData] = None,
    seed: int = 42,
    test_size: float = 0.20,
    val_size_within_train: float = 0.20,
) -> IsoVAEPreprocessor:
    """Reconstruct preprocessing metadata for older checkpoints.

    The final manuscript checkpoint stores feature names and model weights, but
    not the fitted ``StandardScaler``. This helper rebuilds the scaler from the
    original paired training data using the same deterministic split. If
    ``adata_iso_train`` is provided, cells are first aligned exactly as in
    training.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    genes = list(map(str, ckpt["genes_final"]))
    isoforms = list(map(str, ckpt["isoforms_final"]))
    isoform_gene = list(map(str, ckpt["isoform_gene"]))
    groups = [np.asarray(g, dtype=np.int64) for g in ckpt["gene_groups"]]

    if adata_iso_train is not None:
        adata_gene_train, _ = align_paired_cells(adata_gene_train, adata_iso_train)
    adata_gene_train = make_unique_gene_symbol_view(adata_gene_train)
    symbols = adata_gene_train.var["gene_symbol"].astype(str).values
    symbol_to_pos = {g: i for i, g in enumerate(symbols)}
    cols = []
    for gene in genes:
        if gene in symbol_to_pos:
            cols.append(adata_gene_train.X[:, symbol_to_pos[gene]])
        else:
            cols.append(sp.csr_matrix((adata_gene_train.n_obs, 1), dtype=np.float32))
    x_raw = sp.hstack(cols, format="csr") if sp.issparse(cols[0]) else np.column_stack(cols)
    x_norm = normalize_gene_counts(x_raw)

    all_idx = np.arange(adata_gene_train.n_obs)
    train_idx, _ = train_test_split(all_idx, test_size=test_size, random_state=seed)
    train_idx, _ = train_test_split(
        train_idx, test_size=val_size_within_train, random_state=seed
    )
    scaler = StandardScaler().fit(x_norm[train_idx])
    return IsoVAEPreprocessor(
        genes=genes,
        isoforms=isoforms,
        isoform_gene=isoform_gene,
        gene_groups=groups,
        scaler=scaler,
    )


@torch.no_grad()
def predict_usage_array(
    model: IsoVAEModel,
    x: Array,
    groups: Sequence[Array],
    device: str = "cpu",
    batch_size: int = 512,
) -> Array:
    """Predict isoform usage from a scaled gene-expression matrix."""
    model.eval()
    preds: List[Array] = []
    for start in range(0, x.shape[0], batch_size):
        xb = torch.from_numpy(x[start : start + batch_size].astype(np.float32)).to(device)
        logits, _, _ = model.predict_from_gene(xb)
        preds.append(logits_to_usage(logits, groups))
    return np.vstack(preds)


@torch.no_grad()
def denoise_usage_array(
    model: IsoVAEModel,
    y_counts: Array,
    groups: Sequence[Array],
    device: str = "cpu",
    keep_rate: Optional[float] = None,
    batch_size: int = 512,
) -> Tuple[Array, Array]:
    """Denoise long-read isoform counts into within-gene isoform usage.

    If ``keep_rate`` is provided, counts are first binomially downsampled. This
    is useful for simulated denoising experiments.
    """
    model.eval()
    group_tensors = [torch.as_tensor(g, dtype=torch.long, device=device) for g in groups]
    preds: List[Array] = []
    noisy_counts: List[Array] = []
    for start in range(0, y_counts.shape[0], batch_size):
        yc = torch.from_numpy(y_counts[start : start + batch_size].astype(np.float32)).to(device)
        if keep_rate is None:
            yn = yc
        else:
            yn = torch.binomial(torch.clamp(yc, min=0.0), torch.full_like(yc, float(keep_rate)))
        iso_in = iso_encoder_input_from_counts(yn, group_tensors)
        logits, _, _ = model.denoise_from_iso(iso_in)
        preds.append(logits_to_usage(logits, groups))
        noisy_counts.append(yn.detach().cpu().numpy().astype(np.float32))
    return np.vstack(preds), np.vstack(noisy_counts)


def predict_isoform_usage(
    artifact: IsoVAEArtifact,
    adata_gene: ad.AnnData,
    batch_size: int = 512,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Predict isoform usage from short-read-only AnnData."""
    if artifact.preprocessor is None:
        raise ValueError(
            "A fitted IsoVAEPreprocessor is required for AnnData prediction. "
            "Reconstruct it from the paired training data or load one saved with your model."
        )
    x, n_found = artifact.preprocessor.transform_gene_adata(adata_gene)
    pred = predict_usage_array(
        artifact.model, x, artifact.gene_groups, device=artifact.device, batch_size=batch_size
    )
    df = pd.DataFrame(pred, index=adata_gene.obs_names.astype(str), columns=artifact.isoforms)
    return df, {"n_gene_features_found": int(n_found), "n_gene_features_total": len(artifact.genes)}


def denoise_isoform_usage(
    artifact: IsoVAEArtifact,
    adata_iso: ad.AnnData,
    keep_rate: Optional[float] = None,
    batch_size: int = 512,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Denoise long-read isoform counts from AnnData.

    Returns denoised usage, direct observed/noisy usage, and a metadata dict.
    """
    if artifact.preprocessor is not None:
        counts, n_found = artifact.preprocessor.extract_iso_counts(adata_iso)
    else:
        # Feature extraction can still be done from checkpoint metadata.
        pre = IsoVAEPreprocessor(
            genes=artifact.genes,
            isoforms=artifact.isoforms,
            isoform_gene=artifact.isoform_gene,
            gene_groups=artifact.gene_groups,
            scaler=None,  # type: ignore[arg-type]
        )
        counts, n_found = pre.extract_iso_counts(adata_iso)

    denoised, noisy_counts = denoise_usage_array(
        artifact.model,
        counts,
        artifact.gene_groups,
        device=artifact.device,
        keep_rate=keep_rate,
        batch_size=batch_size,
    )
    observed_usage, _, _ = counts_to_gene_usage(noisy_counts, artifact.isoform_gene)
    den_df = pd.DataFrame(denoised, index=adata_iso.obs_names.astype(str), columns=artifact.isoforms)
    obs_df = pd.DataFrame(observed_usage, index=adata_iso.obs_names.astype(str), columns=artifact.isoforms)
    return den_df, obs_df, {"n_isoforms_found": int(n_found), "n_isoforms_total": len(artifact.isoforms)}


def usage_long_table(
    usage: pd.DataFrame,
    isoform_gene: Sequence[str],
    obs_columns: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Convert a cell-by-isoform usage matrix to a long table for plotting."""
    long = usage.reset_index(names="cell").melt(
        id_vars="cell", var_name="isoform", value_name="usage"
    )
    gene_map = pd.DataFrame({"isoform": usage.columns.astype(str), "gene": list(isoform_gene)})
    long = long.merge(gene_map, on="isoform", how="left")
    if obs_columns is not None:
        meta = obs_columns.copy()
        meta["cell"] = meta.index.astype(str)
        long = long.merge(meta, on="cell", how="left")
    return long
