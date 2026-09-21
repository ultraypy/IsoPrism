# Benchmark reproduction

## Scope

The release exposes the eight methods in Supplementary Table S2. TabResNet, FT-Transformer, BABEL and scButterfly are **task-adapted implementations**, not claims of unmodified end-to-end reproduction. Their upstream source revisions are fixed in `short2long/published_models.py`; run `python -m isoprism.upstream` after an editable installation. Upstream licenses remain with those repositories.

The runner does not add Mean, PCA-Ridge, IsoBudget-v2, scHELIX or exploratory spatial models to this benchmark. Internal utility modules retain historical names for compatibility.

## Partitions

| Scenario | Source | Held-out target | Output catalogue |
|---|---|---|---|
| Cross-donor | CRC donors other than PS018 | PS018 | Isoforms |
| Cross-dataset | CRC PacBio resource | FLAMES ONT cell-line resource | Harmonized intron chains |
| Normal-to-tumor | Adjacent-normal CRC tissue | Tumor tissue | Isoforms |
| Cross-platform | FLAMES ONT resource | CRC PacBio resource | Harmonized intron chains |
| Spatial | Source post-MI sections | Section D | Isoforms |

The cross-study directions share a study pair; platform, tissue and study are confounded. Normal-to-tumor is tissue-state transfer, not an independent disease cohort. A single donor holdout is not leave-every-donor-out evaluation.

Catalogues use **source LR support only**: each target has at least 3 counts across at least 2 source cells, and each retained gene has at least 2 retained targets. Up to 2,000 input genes are selected on the source partition. Source-only inner partitions select hyperparameters; the final model is refitted on all source observations. Test LR counts are held separately and opened by the evaluator, not by the prediction model.

SR-only pseudoclusters with target size 50 provide local context. These differ from the 10 fixed SR evaluation clusters. Annotation-based queries do not use target LR expression. Harmonized-chain scenarios use an unavailable-endpoint mask rather than borrowing transcript endpoints.

## Commands

```bash
isoprism-prepare --config configs/benchmark.json --data-root /path/to/project --output-root runs/prepared
isoprism-benchmark --prepared-root runs/prepared --output-root runs/benchmark \
  --config configs/benchmark.json --data-root /path/to/project --check-only
isoprism-benchmark --prepared-root runs/prepared --output-root runs/benchmark \
  --config configs/benchmark.json --data-root /path/to/project
```

The full plan has 110 runs: 3 seeds for each of 7 neural methods plus one deterministic PCA-kNN fit, in each of 5 settings. Use `--methods IsoPrism` for the 15 IsoPrism runs. Defaults are seeds 2026/2027/2028, learning rates 0.0003/0.001, at most 50 epochs and patience 8. IsoPrism uses 8 programs, 4 heads, 64-dimensional tokens, 4 allocation steps and cluster loss weight 0.5. Each method selects its setting using source-validation JSD. Baseline-specific batch sizes and adaptation losses are preserved in the original implementation.

Resume with the same command and output directory. Changed code, configuration or input hashes require a new output directory. Never overwrite historical results. Set `CUDA_VISIBLE_DEVICES` to select a physical GPU; the runner uses logical `cuda:0`. No new training is performed by installation.

## Metrics

All scores concern within-gene proportions, not absolute transcript abundance.

- **Dominant isoform accuracy**: average accuracy within each eligible gene, then average equally across genes. Observed top proportion must be at least 0.5 with a top-two margin of at least 0.1.
- **Cell-gene 1-JSD**: one minus the average base-2 Jensen-Shannon divergence across cell/spot-gene pairs with at least 2 observed LR counts.
- **Cluster allocation Spearman**: average observed and predicted proportions over the same eligible cells within each fixed SR cluster, concatenate the retained isoform fractions, then calculate Spearman correlation. Every defined cluster score is exported. The requested maximum is named `cluster_spearman_max`; it is a best-cluster statistic, not overall performance. The median is exported separately.
- **Spatial Moran Spearman**: optional correlation between observed and predicted isoform Moran statistics on the fixed source-selected panel. Undefined predictions remain missing, with coverage retained in each run's JSON.

`summary.csv` reports the mean and sample SD over defined runs, with the number of defined and completed runs. Seeds measure optimization variability, not biological replication. Failed runs are recorded explicitly and stop the queue.

## Version provenance

Current **IsoPrism = Competitive-TA**. Historical **IsoVAE** and **IsoBudget-v2** are distinct architectures. This source release does not relabel historical results or claim that renaming a method reproduces existing spreadsheet values. Original spreadsheets are not modified or redistributed. Generate tables from the intended model's run files, recorded input hashes and explicit metric definitions.

`source_provenance.json` records the input source-file hashes before publication changes. Tests check the scientific core, normalization, input alignment and report aggregation. Full five-setting retraining is separate from package verification.
