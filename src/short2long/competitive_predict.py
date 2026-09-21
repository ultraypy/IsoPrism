"""Short-read-only inference for IsoPrism (formerly Competitive-TA) checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch

from .competitive_model import CompetitiveAllocationNet
from .data import normalize_log_cpm
from .grouping import build_pseudocluster_labels, group_means


def align_genes(source, target):
    exact = {str(gene): i for i, gene in enumerate(source)}
    if len(exact) != len(source):
        raise ValueError("Duplicate input gene identifiers")
    canonical = lambda gene: re.sub(r"^(ENS[A-Z]*G\d+)\.\d+$", r"\1", str(gene))
    fallback = {}
    for i, gene in enumerate(source):
        fallback.setdefault(canonical(gene), []).append(i)
    rows, columns = [], []
    for j, gene in enumerate(target):
        choices = [exact[str(gene)]] if str(gene) in exact else fallback.get(canonical(gene), [])
        if len(choices) > 1:
            raise ValueError(f"Ambiguous input gene: {gene}")
        if choices:
            rows.append(choices[0])
            columns.append(j)
    return sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, columns)), shape=(len(source), len(target)))


def infer_h5ad(checkpoint_path, input_path, output_path, device="cuda:0", microbatch=16):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("Use a new output file; existing data are preserved")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if microbatch < 1:
        raise ValueError("Microbatch must be positive")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = CompetitiveAllocationNet(**saved["model_config"]).to(device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    catalogue, preprocessing = saved["catalogue"], saved["preprocessing"]
    data = ad.read_h5ad(input_path)
    if not data.n_obs or data.obs_names.has_duplicates:
        raise ValueError("Input must have nonempty, unique cell identifiers")
    counts = sparse.csr_matrix(data.X, dtype=np.float32)
    if not np.isfinite(counts.data).all() or (counts.data < 0).any() or (np.abs(counts.data - np.round(counts.data)) > 1e-3).any():
        raise ValueError("Input X must contain finite nonnegative raw integer counts")
    selector = align_genes(data.var_names, catalogue["input_genes"])
    if not selector.nnz:
        raise ValueError("No expected input genes matched")
    x = normalize_log_cpm((counts @ selector).tocsr())
    labels, _ = build_pseudocluster_labels(x, seed=preprocessing["context_seed"], target_size=preprocessing["context_target_size"])
    context_means = group_means(x, labels)
    context = context_means[labels]
    predictions = np.empty((len(x), model.n_isoforms), dtype=np.float32)
    cluster_predictions = np.empty((len(context_means), model.n_isoforms), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), microbatch):
            ix = slice(start, start + microbatch)
            predictions[ix] = model(torch.tensor(x[ix], device=device), torch.tensor(context[ix], device=device)).cpu().numpy()
        for start in range(0, len(context_means), microbatch):
            ix = slice(start, start + microbatch)
            cx = torch.tensor(context_means[ix], device=device)
            cluster_predictions[ix] = model(cx, cx).cpu().numpy()
    obs = data.obs.copy()
    obs["isoprism_context_id"] = labels
    obs["competitive_context_id"] = labels  # Legacy column alias.
    groups = np.asarray(catalogue["groups"])
    var = pd.DataFrame({"parent_gene_id": np.asarray(catalogue["parents"])[groups],
                        "parent_gene_index": groups, "structure_id": catalogue["structures"]},
                       index=pd.Index(catalogue["isoforms"], name="isoform_id"))
    output = ad.AnnData(predictions, obs=obs, var=var, layers={"context_proportions": cluster_predictions[labels]})
    output.uns["competitive_allocation"] = {"value": "within-gene isoform proportion", "iterations": model.steps,
        "matched_input_genes": selector.nnz, "expected_input_genes": len(catalogue["input_genes"]),
        "missing_genes": "zero-filled; extensive missingness requires independent validation",
        "context": "SR-only and dependent on the supplied cell collection; not the fixed evaluation clusters",
        "query_mode": model.query_mode, "no_empirical_allocation_prior": True,
        "normalization": preprocessing["normalization"]}
    output.uns["isoprism"] = dict(output.uns["competitive_allocation"], method="IsoPrism")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_h5ad(output_path, compression="gzip")
    return {"cells_or_spots": len(x), "isoforms": model.n_isoforms, "matched_input_genes": selector.nnz,
            "expected_input_genes": len(catalogue["input_genes"]), "context_clusters": len(context_means)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-h5ad", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--microbatch", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(infer_h5ad(args.model, args.input_h5ad, args.output_h5ad, args.device, args.microbatch))


if __name__ == "__main__":
    main()
