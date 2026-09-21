"""IsoPrism: short-read expression to within-gene isoform proportions.

The current method is the terminal-aware Competitive-TA implementation.
It is not the historical IsoVAE or IsoBudget-v2 model.
"""
from short2long.competitive_model import IsoPrism
from short2long.competitive_predict import infer_h5ad

__version__ = "0.2.0"
__all__ = ["IsoPrism", "infer_h5ad"]
