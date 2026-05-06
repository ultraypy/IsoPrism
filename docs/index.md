# IsoVAE

**IsoVAE** is a Python package for single-cell isoform-usage analysis. It supports two main tasks:

1. **Isoform-usage prediction** from short-read single-cell gene-expression profiles.
2. **Long-read isoform-usage denoising** from sparse long-read isoform count matrices.

IsoVAE models **within-gene isoform usage proportions**, not absolute transcript abundance.

## What IsoVAE takes as input

| Task | Input | Output |
|---|---|---|
| Prediction | short-read gene-expression AnnData | isoform-usage matrix |
| Denoising | long-read isoform-count AnnData | denoised isoform-usage matrix |

## Typical use cases

- Infer candidate isoform-usage patterns in short-read-only scRNA-seq datasets.
- Denoise sparse long-read single-cell isoform measurements.
- Compare isoform usage across cell states, developmental stages, or conditions.

## Minimal example

```python
import scanpy as sc
from isovae import load_artifact, predict_isoform_usage

artifact = load_artifact("path/to/model.pt", preprocessor=preprocessor, device="cpu")
gene_query = sc.read("query_gene_matrix.h5ad")
usage, meta = predict_isoform_usage(artifact, gene_query)
usage.to_csv("predicted_isoform_usage.csv")
```

## Important interpretation note

IsoVAE outputs **gene-wise isoform proportions**. For each gene, the predicted isoform usages sum to 1 across modeled isoforms of that gene. The output should not be interpreted as absolute transcript counts.
