# IsoPrism

IsoPrism predicts **within-gene isoform proportions** for individual cells or spots from short-read gene-expression matrices. It combines short-read local context, program attention and terminal-aware isoform queries with iterative within-gene allocation. Paired long-read counts supervise training; prediction requires only a short-read matrix and a trained checkpoint. No sample FASTQ/BAM is required.

This is the method previously called **Competitive-TA**. The older **IsoVAE** and **IsoBudget-v2** models are different methods, not alternative names for this implementation.

## Install

Use Python 3.11+ and install PyTorch for your CPU/CUDA environment, then:

```bash
git clone https://github.com/ultraypy/IsoPrism.git
cd IsoPrism
pip install -e ".[dev]"
```

Install from GitHub. This release does not publish or update a PyPI package.

## Predict

```bash
isoprism-predict --model model.pt --input-h5ad short_reads.h5ad \
  --output-h5ad isoform_proportions.h5ad --device cpu
```

Input `X` must contain finite, nonnegative raw gene counts, with unique cell/spot and gene identifiers. Output `X` contains isoform proportions summing to one **within each gene**. `layers['context_proportions']` provides a separate local-context readout. Historical Competitive-TA checkpoints and import names remain supported.

## Benchmark

Five scenarios are supported: cross-donor, cross-dataset, normal-to-tumor, cross-platform and spatial transfer. The public runner includes exactly the eight methods in the supplied benchmark table: **PCA-kNN, MLP, Context-MLP, TabResNet, FT-Transformer, BABEL, scButterfly and IsoPrism**.

```bash
# Download pinned comparator code, not sequencing data.
python -m isoprism.upstream

# After preparing local paired matrices, edit configs/benchmark.json as needed.
isoprism-prepare --config configs/benchmark.json --data-root /path/to/project \
  --output-root runs/prepared

isoprism-benchmark --prepared-root runs/prepared --output-root runs/benchmark \
  --config configs/benchmark.json --data-root /path/to/project
```

Use `--methods IsoPrism` to train only our method, `--dry-run` to print the plan, or `--check-only` to verify inputs without training. Benchmark training requires CUDA; prediction and unit tests support CPU.

Results include `individual_runs.csv`, `summary.csv` and `cluster_points.csv`. Both the **maximum** and **median** cluster Spearman are explicitly named; every defined cluster point is retained. Missing scores remain `NA`.

See [benchmark protocol](docs/benchmark.md) and [data format](docs/data-format.md) for details. Raw matrices, pretrained weights and prediction arrays are not included.

## Tests and legacy code

```bash
pytest -q tests/test_competitive_model.py tests/test_competitive_structure.py \
  tests/test_competitive_predict.py tests/test_unified_benchmark.py tests/test_isoprism_release.py
```

`short2long` is the compatibility implementation namespace; the public interface is `isoprism`. Old IsoVAE source remains in `src/isovae` and [archived documentation](docs/legacy/overview.md), but is not installed by this package. Historical result files are not relabelled or overwritten by this release.


