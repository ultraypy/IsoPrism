# Quick start

This page shows the core IsoVAE workflow.

## 1. Load a trained model

```python
import scanpy as sc
from isovae import (
    load_artifact,
    reconstruct_preprocessor_from_training_data,
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
```

## 2. Predict isoform usage from short-read data

```python
from isovae import predict_isoform_usage

gene_query = sc.read("path/to/query_gene_matrix.h5ad")

pred_usage, meta = predict_isoform_usage(
    artifact,
    gene_query,
    batch_size=512,
)

print(meta)
pred_usage.to_csv("predicted_isoform_usage.csv")
```

The returned `pred_usage` is a cell-by-isoform DataFrame.

## 3. Denoise sparse long-read isoform counts

```python
from isovae import denoise_isoform_usage

iso_query = sc.read("path/to/query_isoform_matrix.h5ad")

denoised_usage, noisy_usage, meta = denoise_isoform_usage(
    artifact,
    iso_query,
    keep_rate=None,
    batch_size=512,
)

denoised_usage.to_csv("denoised_isoform_usage.csv")
```

For a simulated low-coverage experiment, set `keep_rate`:

```python
denoised_usage, noisy_usage, meta = denoise_isoform_usage(
    artifact,
    iso_query,
    keep_rate=0.2,
)
```

## 4. Convert usage to a long table

```python
from isovae.inference import usage_long_table

long_df = usage_long_table(
    pred_usage,
    artifact.isoform_gene,
    obs_columns=gene_query.obs,
)

long_df.head()
```

This is convenient for plotting gene-level isoform usage distributions.
