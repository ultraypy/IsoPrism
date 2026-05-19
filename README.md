# IsoVAE

IsoVAE is a Python package for single-cell isoform-usage analysis. It supports:

1. **Isoform-usage prediction** from short-read single-cell gene-expression profiles.
2. **Long-read isoform-usage denoising** from sparse long-read isoform count matrices.

IsoVAE models **within-gene isoform usage proportions**, not absolute transcript abundance.

## Installation

```bash
pip install isovae
```

For local development:

```bash
git clone https://github.com/ultraypy/IsoVAE.git
cd IsoVAE
pip install -e .
```

## Quick start

```python
import scanpy as sc
from isovae import (
    load_artifact,
    reconstruct_preprocessor_from_training_data,
    predict_isoform_usage,
    denoise_isoform_usage,
)

model_path = "path/to/vae_xda_model.pt"

gene_train = sc.read("path/to/training_gene_matrix.h5ad")
iso_train = sc.read("path/to/training_isoform_matrix.h5ad")

preprocessor = reconstruct_preprocessor_from_training_data(
    model_path,
    adata_gene_train=gene_train,
    adata_iso_train=iso_train,
    seed=42,
)

artifact = load_artifact(model_path, preprocessor=preprocessor, device="cpu")

# Predict isoform usage from short-read data.
gene_query = sc.read("path/to/query_gene_matrix.h5ad")
pred_usage, pred_meta = predict_isoform_usage(artifact, gene_query)
pred_usage.to_csv("predicted_isoform_usage.csv")

# Denoise long-read isoform usage.
iso_query = sc.read("path/to/query_isoform_matrix.h5ad")
denoised_usage, noisy_usage, denoise_meta = denoise_isoform_usage(artifact, iso_query)
denoised_usage.to_csv("denoised_isoform_usage.csv")
```


