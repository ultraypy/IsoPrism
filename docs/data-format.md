# Data format

IsoVAE works with `AnnData` objects.

## Short-read gene-expression matrix

The short-read input should be an AnnData object with:

- cells in `.obs_names`;
- genes in `.var_names` or gene symbols available in `.var["gene_symbol"]`;
- raw or count-like expression values in `.X`.

Example:

```python
import scanpy as sc
adata_gene = sc.read("gene_matrix.h5ad")
```

## Long-read isoform-count matrix

The long-read input should be an AnnData object with:

- cells in `.obs_names`;
- isoforms in `.var_names`;
- isoform counts in `.X`.

Example:

```python
adata_iso = sc.read("isoform_matrix.h5ad")
```

## Paired training data

For reconstructing preprocessing from an existing checkpoint, paired short-read and long-read cells should share cell identifiers. IsoVAE aligns paired cells internally.

```python
from isovae import reconstruct_preprocessor_from_training_data

preprocessor = reconstruct_preprocessor_from_training_data(
    checkpoint="path/to/model.pt",
    adata_gene_train=adata_gene,
    adata_iso_train=adata_iso,
)
```

## Output format

IsoVAE returns a cell-by-isoform `pandas.DataFrame`. Values are isoform-usage proportions within each gene.

For a modeled gene with multiple isoforms, the usage values across those isoforms sum to approximately 1 for each cell.
