"""Predict IsoPrism proportions from a short-read H5AD and a checkpoint."""
from short2long.competitive_predict import align_genes, infer_h5ad, main

if __name__ == "__main__":
    main()
