# FAQ

## Does IsoVAE predict absolute isoform expression?

No. IsoVAE predicts or denoises **within-gene isoform usage proportions**.

## Can I use IsoVAE without long-read data?

Yes, for prediction. Once a trained IsoVAE model is available, it can be applied to short-read-only gene-expression matrices to infer candidate isoform-usage patterns.

## Do I need paired short-read and long-read data?

Paired data are needed to train the model or reconstruct preprocessing for older checkpoints. For downstream prediction, a trained model can be applied to short-read-only data.

## Why do I need the preprocessor?

The preprocessor stores feature alignment and scaling information. It ensures that new input matrices are transformed into the same feature space used during model training.

## What should I upload to GitHub?

For a clean package repository, upload:

```text
src/
docs/
mkdocs.yml
pyproject.toml
README.md
requirements.txt
LICENSE
.gitignore
```

Do not upload large data files, model checkpoints, manuscript drafts or generated result folders.
