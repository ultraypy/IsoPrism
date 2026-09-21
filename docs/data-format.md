# Data format

## Prediction

Supply an AnnData `.h5ad` with cells/spots in rows, genes in columns, unique identifiers and raw nonnegative integer counts in `X`. Counts are aligned to checkpoint genes, normalized to 10,000 counts over matched selected genes and log1p transformed. Missing genes are zero-filled and their number is reported. Extensive missingness requires independent validation. Do not pass already log-normalized data as raw counts.

The output is cell/spot by isoform, with gene identities in `var['parent_gene_id']`. Proportions sum to one within each retained gene. They do not sum to one across the entire transcriptome and are not absolute expression estimates.

## Paired matrices for training and evaluation

Each processed dataset directory contains:

| File | Content |
|---|---|
| `x_gene_counts.npz` | SciPy sparse CSR cell-by-gene nonnegative matrix |
| `y_isoform_counts.npz` | SciPy sparse CSR cell-by-isoform LR count matrix |
| `obs.csv` | `cell_id` index and split columns such as donor, condition, domain or section |
| `genes.csv` | `gene_id` column, matching X columns |
| `isoforms.csv` | `isoform_id`, `gene_id`, `structure_id`, matching Y columns |
| `dataset.json` | Dataset provenance and preprocessing metadata |

X and Y must have the same cell order. Spatial data also require `spatial_x` and `spatial_y` in `obs.csv`. Splice-chain identifiers use `gene|chromosome|strand|donor-acceptor;donor-acceptor` or the formats parsed by `short2long.structure`.

`scripts/prepare_crc.py`, `prepare_spatial.py` and `prepare_cross_platform_v2.py` adapt the original processed public releases; inspect their `--help`. Cross-platform preparation uses the author-release barcode mapping and complete intron-chain harmonization. No FASTQ alignment is involved. These scripts require files obtained separately from the original data providers.

The historical CRC resource supplies normalized SR expression. Its preparation script converts from log scale as recorded by that resource; those values are not recovered raw UMI counts. Preserve that provenance when reproducing historical experiments. The strict raw-count prediction interface is a separate deployment contract.

Edit `configs/benchmark.json` to point to these local datasets and annotations, relative to `--data-root`. The preparation command writes inner/final catalogues, matrices, SR-only contexts, fixed evaluation clusters, separate test truth and SHA-256 manifests. Benchmark methods share these files without changing them. Annotation files and all biological matrices stay outside Git.
