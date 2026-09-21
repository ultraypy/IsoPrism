"""Mean-anchored state corrections using the existing program attention encoder."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .model import IsoBudgetAttentionNet, grouped_softmax


class ConservativeIsoBudget(IsoBudgetAttentionNet):
    def __init__(self, n_genes, groups, structures, prior, reference,
                 d_model=64, n_programs=8, n_heads=4, dropout=0.1):
        groups = torch.as_tensor(groups, dtype=torch.long)
        super().__init__(n_genes, len(groups), groups, structures,
                         d_model=d_model, n_programs=n_programs, n_heads=n_heads, dropout=dropout)
        prior = torch.as_tensor(prior, dtype=torch.float32)
        self.register_buffer("prior", prior)
        self.register_buffer("reference_expression", torch.as_tensor(reference, dtype=torch.float32))
        self.register_buffer("calibration_alpha", torch.tensor(1.0))
        self.cell_gate_logits = nn.Parameter(torch.full((self.n_target_genes,), -2.0))
        # The first prediction is exactly the source prior, not a random allocation.
        nn.init.zeros_(self.score_scale)
        self.log_concentration.requires_grad_(False)  # No DM head in this version.

    def forward(self, x, context=None, return_components=False, calibrated=True):
        if context is None:
            context = x
        n = len(x)
        programs, _ = self._encode_programs(
            torch.cat((x, context, self.reference_expression[None, :])), need_weights=False)
        scores, _ = self._scores_from_programs(programs)
        cell, cluster, reference = scores[:n], scores[n:2*n], scores[-1:]
        gates = self.cell_gate_logits.sigmoid()[self.isoform_gene_index][None, :]
        cluster_delta = cluster - reference
        cell_delta = cluster_delta + gates * (cell - cluster)
        # A bounded log-ratio correction: no unconstrained cluster-level offset.
        cluster_delta = 2.0 * torch.tanh(cluster_delta / 2.0)
        cell_delta = 2.0 * torch.tanh(cell_delta / 2.0)
        p = grouped_softmax(self.prior.log()[None, :] + cell_delta,
                            self.isoform_gene_index, self.n_target_genes)
        cp = grouped_softmax(self.prior.log()[None, :] + cluster_delta,
                             self.isoform_gene_index, self.n_target_genes)
        if calibrated:
            alpha = self.calibration_alpha
            p = (1 - alpha) * self.prior + alpha * p
            cp = (1 - alpha) * self.prior + alpha * cp
        return (p, cp) if return_components else p


def totals_by_gene(y, groups, n_groups):
    return y.new_zeros((len(y), n_groups)).scatter_add_(1, groups[None, :].expand(len(y), -1), y)


def cell_targets(y, groups, prior, prior_strength=2.0):
    totals = totals_by_gene(y, groups, int(groups.max()) + 1)
    valid = totals >= 2
    targets = (y + prior_strength * prior[None, :]) / (
        totals[:, groups] + prior_strength).clamp_min(1e-8)
    return targets, valid


def macro_cross_entropy(probabilities, targets, valid, groups):
    numerator = -(targets * probabilities.clamp_min(1e-8).log() * valid[:, groups]).sum()
    return numerator / valid.sum().clamp_min(1)


def macro_jsd(probabilities, counts, groups):
    totals = totals_by_gene(counts, groups, int(groups.max()) + 1)
    valid = totals >= 2
    truth = counts / totals[:, groups].clamp_min(1e-8)
    midpoint = (truth + probabilities) / 2
    divergence = 0.5 * (torch.special.xlogy(truth, truth / midpoint.clamp_min(1e-8))
                        + torch.special.xlogy(probabilities, probabilities / midpoint.clamp_min(1e-8)))
    return (divergence * valid[:, groups]).sum() / np.log(2), valid.sum()


def choose_alpha(scores):
    """Use a prespecified grid, preferring less correction on numerical ties."""
    best = min(row["macro_jsd"] for row in scores)
    return min(row["alpha"] for row in scores if row["macro_jsd"] <= best + 1e-8)


def cluster_contrast_metrics(counts, probabilities, groups, labels, min_cells=5):
    """Gene-macro MAE of all eligible pairwise cluster proportion differences.

    Eligibility and cell sets are shared for truth and predictions, fixed at
    parent count >= 2 and at least five valid cells per cluster-gene.
    This is a diagnostic on an already inspected test set, not a tuning target.
    """
    if hasattr(counts, "toarray"):
        counts = counts.toarray()
    gene_errors, comparisons = [], 0
    for group in np.unique(groups):
        ix = np.flatnonzero(groups == group)
        totals = counts[:, ix].sum(1)
        observed, predicted = [], []
        for label in np.unique(labels):
            valid = (labels == label) & (totals >= 2)
            if valid.sum() >= min_cells:
                observed.append((counts[valid][:, ix] / totals[valid, None]).mean(0))
                predicted.append(probabilities[valid][:, ix].mean(0))
        errors = []
        for left in range(len(observed)):
            for right in range(left + 1, len(observed)):
                errors.append(np.abs((predicted[left] - predicted[right]) -
                                     (observed[left] - observed[right])).mean())
        if errors:
            gene_errors.append(np.mean(errors))
            comparisons += len(errors)
    return {"cluster_contrast_mae": float(np.mean(gene_errors)) if gene_errors else None,
            "contrast_evaluable_genes": len(gene_errors), "contrast_cluster_gene_pairs": comparisons,
            "contrast_min_cells_per_cluster_gene": min_cells}
