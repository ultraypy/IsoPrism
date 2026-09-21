"""State-dependent, structure-guided allocation on a product of gene simplices.

No pooled allocation prior, reference-expression subtraction, calibration mixture,
or prior-shrinkage objective is used. Iterations are computational, not biological
time. The learned update field has no claimed convergence or causal guarantee.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn

from .model import grouped_softmax, splice_structure_tensors


class IsoPrism(nn.Module):
    def __init__(self, n_input, groups, structures, d_model=64, n_programs=8,
                 n_heads=4, steps=4, dropout=0.1, query_mode="structure", horizon=1.0,
                 terminal_features=None):
        super().__init__()
        groups = torch.as_tensor(groups, dtype=torch.long)
        if groups.ndim != 1 or not len(groups) or (groups < 0).any():
            raise ValueError("Nonempty nonnegative isoform-to-gene groups are required")
        if not torch.equal(groups.unique(), torch.arange(int(groups.max()) + 1)):
            raise ValueError("Parent-gene group identifiers must be contiguous")
        if len(groups) != len(structures) or steps < 1 or d_model % n_heads:
            raise ValueError("Invalid structure catalogue, iteration count, or head dimensions")
        if query_mode not in {"structure", "identity"} or horizon <= 0:
            raise ValueError("Invalid query representation or integration horizon")
        self.n_target_genes = int(groups.max()) + 1
        self.n_isoforms, self.n_programs, self.d_model = len(groups), n_programs, d_model
        self.steps, self.horizon, self.query_mode = steps, float(horizon), query_mode
        self.register_buffer("isoform_gene_index", groups)
        self.register_buffer("isoforms_per_gene", torch.bincount(groups).float())
        self.gene_embedding = nn.Embedding(n_input, d_model)
        # Cell expression and SR-only local context enter the SAME token encoder.
        self.expression_projection = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.program_queries = nn.Parameter(torch.empty(n_programs, d_model))
        nn.init.normal_(self.program_queries, std=0.02)
        self.program_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.program_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.parent_embedding = nn.Embedding(self.n_target_genes, d_model)
        if query_mode == "structure":
            junctions, offsets, features = splice_structure_tensors(structures)
            if terminal_features is not None:
                terminal = torch.as_tensor(terminal_features, dtype=torch.float32)
                if terminal.shape != (len(groups), 5) or not torch.isfinite(terminal).all():
                    raise ValueError("Five finite terminal descriptors are required per isoform")
                # Extend the EXISTING structural projection, not an extra network.
                # Endpoint information prevents identical-chain endpoint variants
                # from being mathematically forced to equal allocations.
                features = torch.cat((features, terminal), dim=1)
            self.register_buffer("junction_ids", junctions)
            self.register_buffer("junction_offsets", offsets)
            self.register_buffer("structure_features", features)
            self.junction_embedding = nn.EmbeddingBag(int(junctions.max()) + 1, d_model, mode="mean", padding_idx=0)
            self.structure_projection = nn.Sequential(nn.Linear(features.shape[1], d_model), nn.GELU(), nn.Linear(d_model, d_model))
        else:
            # Removing structural features must not make within-gene queries identical.
            self.isoform_embedding = nn.Embedding(self.n_isoforms, d_model)
        self.isoform_norm = nn.LayerNorm(d_model)
        # One shared field for every step. The state contains the current allocation
        # and the difference from the gene's allocation-weighted program usage.
        self.state_feedback = nn.Sequential(nn.Linear(n_programs + 1, 16), nn.Tanh(), nn.Linear(16, n_programs))

    def encode(self, x, context):
        if x.ndim != 2 or x.shape != context.shape or x.shape[1] != self.gene_embedding.num_embeddings:
            raise ValueError("Cell and context expressions must have matching input-gene dimensions")
        ids = torch.arange(x.shape[1], device=x.device)
        tokens = self.gene_embedding(ids)[None] + self.expression_projection(torch.stack((x, context), dim=-1))
        tokens = self.dropout(tokens)
        mask = (x <= 0) & (context <= 0)
        if mask.all(1).any():
            mask = mask.clone()
            mask[mask.all(1), 0] = False
        queries = self.program_queries[None].expand(len(x), -1, -1)
        programs, _ = self.program_attention(queries, tokens, tokens, key_padding_mask=mask, need_weights=False)
        return self.program_norm(queries + self.dropout(programs))

    def isoform_queries(self):
        parent = self.parent_embedding(self.isoform_gene_index)
        if self.query_mode == "identity":
            representation = self.isoform_embedding.weight
        else:
            representation = self.junction_embedding(self.junction_ids, self.junction_offsets)
            representation = representation + self.structure_projection(self.structure_features)
        return self.isoform_norm(parent + representation)

    def update_rates(self, probabilities, compatibility):
        preference = compatibility.softmax(-1)
        index = self.isoform_gene_index[None, :, None].expand(len(probabilities), -1, self.n_programs)
        expected = probabilities.new_zeros((len(probabilities), self.n_target_genes, self.n_programs))
        expected.scatter_add_(1, index, probabilities[:, :, None] * preference)
        competing_usage = expected.gather(1, index)
        relative_mass = probabilities.clamp_min(1e-12).log() + self.isoforms_per_gene[self.isoform_gene_index].log()[None]
        state = torch.cat((relative_mass[:, :, None], preference - competing_usage), dim=-1)
        feedback = self.state_feedback(state)
        adjusted = compatibility + feedback
        return (adjusted.softmax(-1) * adjusted).sum(-1)

    def forward(self, x, context=None, *, steps=None, frozen_state=False, return_trajectory=False):
        context = x if context is None else context
        programs = self.encode(x, context)
        compatibility = torch.einsum("id,bmd->bim", self.isoform_queries(), programs) / math.sqrt(self.d_model)
        n_steps = self.steps if steps is None else int(steps)
        if n_steps < 1:
            raise ValueError("At least one allocation update is required")
        uniform = (1.0 / self.isoforms_per_gene[self.isoform_gene_index])[None].expand(len(x), -1)
        p = uniform
        trajectory = [p] if return_trajectory else None
        # Equal total horizon prevents iteration-count ablations from merely
        # changing the softmax temperature.
        for _ in range(n_steps):
            rates = self.update_rates(uniform if frozen_state else p, compatibility)
            p = grouped_softmax(p.clamp_min(1e-12).log() + (self.horizon / n_steps) * rates,
                                self.isoform_gene_index, self.n_target_genes)
            if return_trajectory:
                trajectory.append(p)
        return (p, trajectory) if return_trajectory else p


# Historical import name retained for existing experiment scripts/checkpoints.
CompetitiveAllocationNet = IsoPrism


def multinomial_nll_terms(probabilities, counts):
    """Parameter-dependent conditional multinomial NLL and observed molecule count.

    The multinomial coefficient is independent of network parameters and omitted.
    No pseudocounts or empirical allocation prior are added. A molecule-normalized
    objective deliberately weights high-coverage observations more heavily.
    """
    return -(counts * probabilities.clamp_min(1e-12).log()).sum(), counts.sum()


def unsmoothed_cluster_targets(counts, labels, groups, min_count=2, min_cells=3):
    """Equal-cell observed fractions, computed in bounded-memory chunks."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    n_clusters, n_genes = int(labels.max()) + 1, int(groups.max()) + 1
    summed = np.zeros((n_clusters, len(groups)), dtype=np.float32)
    coverage = np.zeros((n_clusters, n_genes), dtype=np.float32)
    for start in range(0, len(counts), 128):
        y = np.asarray(counts[start:start + 128], dtype=np.float32)
        totals = np.zeros((len(y), n_genes), dtype=np.float32)
        np.add.at(totals.T, groups, y.T)
        valid = totals >= min_count
        fractions = y / np.maximum(totals[:, groups], 1)
        np.add.at(summed, labels[start:start + len(y)], fractions * valid[:, groups])
        np.add.at(coverage, labels[start:start + len(y)], valid.astype(np.float32))
    targets = summed / np.maximum(coverage[:, groups], 1)
    return targets, coverage >= min_cells
