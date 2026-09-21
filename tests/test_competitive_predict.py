import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from short2long.competitive_model import CompetitiveAllocationNet
from short2long.competitive_predict import align_genes, infer_h5ad


def test_gene_mapping_is_strict():
    selector = align_genes(["ENSG0001.2", "B", "A"], ["A", "ENSG0001", "missing"])
    np.testing.assert_array_equal(selector.toarray(), [[0, 1, 0], [0, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError):
        align_genes(["A", "A"], ["A"])
    with pytest.raises(ValueError):
        align_genes(["ENSG0001.1", "ENSG0001.2"], ["ENSG0001"])


def test_h5ad_inference_needs_only_gene_counts_and_checkpoint(tmp_path):
    cfg = {"n_input": 3, "groups": [0, 0], "structures": ["g|chr1|+|10-20", "g|chr1|+|10-30"],
           "d_model": 8, "n_programs": 4, "n_heads": 2, "steps": 4, "dropout": 0.}
    model = CompetitiveAllocationNet(**cfg)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_config": cfg, "model_state": model.state_dict(),
                "catalogue": {"input_genes": ["A", "B", "C"], "isoforms": ["i1", "i2"], "parents": ["g"],
                              "groups": cfg["groups"], "structures": cfg["structures"]},
                "preprocessing": {"context_seed": 2026, "context_target_size": 50, "normalization": "log1p CPM"}}, checkpoint)
    source, destination = tmp_path / "short_reads.h5ad", tmp_path / "proportions.h5ad"
    ad.AnnData(np.array([[4, 1, 0], [2, 3, 1], [0, 0, 0]], dtype=np.float32),
               obs=pd.DataFrame(index=["c1", "c2", "c3"]), var=pd.DataFrame(index=["A", "B", "C"])).write_h5ad(source)
    report = infer_h5ad(checkpoint, source, destination, device="cpu", microbatch=2)
    assert report["matched_input_genes"] == 3
    result = ad.read_h5ad(destination)
    assert result.shape == (3, 2)
    np.testing.assert_allclose(result.X.sum(1), 1, atol=1e-6)
    np.testing.assert_allclose(result.layers["context_proportions"].sum(1), 1, atol=1e-6)
    assert result.uns["competitive_allocation"]["no_empirical_allocation_prior"]
    with pytest.raises(FileExistsError):
        infer_h5ad(checkpoint, source, destination, device="cpu")
