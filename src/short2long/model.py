from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


def grouped_softmax(scores: torch.Tensor, group_index: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Apply a numerically stable softmax independently inside every group."""
    groups = group_index.unsqueeze(0).expand(scores.shape[0], -1)
    maxima = torch.full(
        (scores.shape[0], n_groups),
        -torch.inf,
        dtype=scores.dtype,
        device=scores.device,
    )
    maxima.scatter_reduce_(1, groups, scores, reduce="amax", include_self=True)
    shifted = scores - maxima.gather(1, groups)
    exponentials = shifted.exp()
    totals = torch.zeros_like(maxima)
    totals.scatter_add_(1, groups, exponentials)
    return exponentials / totals.gather(1, groups).clamp_min(1e-8)


class IsoformAllocationNet(nn.Module):
    """Small MLP retained as the original feasibility model."""

    def __init__(
        self,
        n_genes: int,
        n_isoforms: int,
        isoform_gene_index: torch.Tensor,
        hidden: int = 256,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_isoforms),
        )
        self.register_buffer("isoform_gene_index", isoform_gene_index.long())
        self.n_target_genes = int(isoform_gene_index.max().item()) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.nn.functional.softplus(self.encoder(x)) + 1e-8
        groups = self.isoform_gene_index.unsqueeze(0).expand(x.shape[0], -1)
        totals = torch.zeros(
            (x.shape[0], self.n_target_genes), device=x.device, dtype=scores.dtype
        )
        totals.scatter_add_(1, groups, scores)
        return scores / totals.gather(1, groups).clamp_min(1e-8)


def _parse_splice_chain(isoform_id: str) -> tuple[str, str, list[tuple[int, int]]]:
    """Parse the exact-splice-chain IDs produced by the harmonization pipeline."""
    parts = str(isoform_id).split("|", 3)
    if len(parts) != 4 or parts[2] not in {"+", "-"}:
        return "unknown", ".", []
    junctions: list[tuple[int, int]] = []
    for item in parts[3].split(";"):
        match = re.fullmatch(r"(\d+)-(\d+)", item)
        if match is None:
            return "unknown", ".", []
        start, end = int(match.group(1)), int(match.group(2))
        if end <= start:
            return "unknown", ".", []
        junctions.append((start, end))
    return parts[1], parts[2], junctions


def splice_structure_tensors(
    isoform_ids: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a junction bag plus continuous splice-chain descriptors.

    Junction vocabulary construction is deterministic, so a checkpoint can be
    reconstructed from its ordered target isoform catalogue during inference.
    """
    parsed = [_parse_splice_chain(value) for value in isoform_ids]
    keys = sorted(
        {
            (chromosome, strand, start, end)
            for chromosome, strand, junctions in parsed
            for start, end in junctions
        }
    )
    vocabulary = {key: i + 1 for i, key in enumerate(keys)}
    flat_junctions: list[int] = []
    offsets: list[int] = []
    features: list[list[float]] = []
    for chromosome, strand, junctions in parsed:
        offsets.append(len(flat_junctions))
        if junctions:
            flat_junctions.extend(
                vocabulary[(chromosome, strand, start, end)] for start, end in junctions
            )
            lengths = np.asarray([end - start for start, end in junctions], dtype=np.float64)
            span = float(junctions[-1][1] - junctions[0][0])
            values = [
                math.log1p(len(junctions)),
                math.log1p(lengths.sum()),
                math.log1p(lengths.mean()),
                math.log1p(lengths.std()),
                math.log1p(lengths.min()),
                math.log1p(lengths.max()),
                math.log1p(max(span, 0.0)),
                1.0 if strand == "+" else -1.0,
                1.0,
            ]
        else:
            # EmbeddingBag needs one entry per isoform. Index zero is padding.
            flat_junctions.append(0)
            values = [0.0] * 9
        features.append(values)
    feature_array = np.asarray(features, dtype=np.float32)
    continuous = feature_array[:, :7]
    valid = feature_array[:, 8] > 0
    if valid.any():
        mean = continuous[valid].mean(axis=0)
        std = continuous[valid].std(axis=0)
        std[std < 1e-5] = 1.0
        continuous[valid] = (continuous[valid] - mean) / std
    feature_array[:, :7] = continuous
    return (
        torch.tensor(flat_junctions, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.from_numpy(feature_array),
    )


class IsoBudgetAttentionNet(nn.Module):
    """Structure-constrained latent splicing-program attention model.

    A small bank of program queries reads the expressed-gene tokens. Candidate
    splice-chain queries then compete for those programs inside each parent gene.
    """

    def __init__(
        self,
        n_genes: int,
        n_isoforms: int,
        isoform_gene_index: torch.Tensor,
        isoform_ids: Sequence[str],
        d_model: int = 128,
        n_programs: int = 16,
        n_heads: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if len(isoform_ids) != n_isoforms:
            raise ValueError("one structure identifier is required for every isoform")
        self.n_target_genes = int(isoform_gene_index.max().item()) + 1
        self.d_model = d_model
        self.n_programs = n_programs
        self.n_heads = n_heads
        self.register_buffer("isoform_gene_index", isoform_gene_index.long())

        junction_ids, junction_offsets, structure_features = splice_structure_tensors(
            isoform_ids
        )
        self.register_buffer("junction_ids", junction_ids)
        self.register_buffer("junction_offsets", junction_offsets)
        self.register_buffer("structure_features", structure_features)
        n_junctions = int(junction_ids.max().item()) if junction_ids.numel() else 0

        self.gene_embedding = nn.Embedding(n_genes, d_model)
        self.expression_projection = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.program_queries = nn.Parameter(torch.empty(n_programs, d_model))
        self.program_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.program_norm = nn.LayerNorm(d_model)

        self.target_gene_embedding = nn.Embedding(self.n_target_genes, d_model)
        self.junction_embedding = nn.EmbeddingBag(
            n_junctions + 1, d_model, mode="mean", padding_idx=0
        )
        self.structure_projection = nn.Sequential(
            nn.Linear(structure_features.shape[1], d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.isoform_norm = nn.LayerNorm(d_model)
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.log_concentration = nn.Parameter(torch.full((self.n_target_genes,), 10.0))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.program_queries, std=0.02)

    def isoform_queries(self, mask_probability: float = 0.0) -> torch.Tensor:
        junction_ids = self.junction_ids
        structure_features = self.structure_features
        if mask_probability > 0:
            junction_mask = (junction_ids != 0) & (
                torch.rand(junction_ids.shape, device=junction_ids.device)
                < mask_probability
            )
            junction_ids = junction_ids.masked_fill(junction_mask, 0)
            feature_mask = torch.rand_like(structure_features) < mask_probability
            feature_mask[:, -1] = False
            structure_features = structure_features.masked_fill(feature_mask, 0)
        junction = self.junction_embedding(junction_ids, self.junction_offsets)
        structure = self.structure_projection(structure_features)
        parent = self.target_gene_embedding(self.isoform_gene_index)
        return self.isoform_norm(parent + junction + structure)

    def structure_consistency_loss(self, mask_probability: float = 0.15) -> torch.Tensor:
        """Keep isoform embeddings stable after junction/descriptor masking."""
        clean = self.isoform_queries().detach()
        corrupted = self.isoform_queries(mask_probability=mask_probability)
        return 1.0 - torch.nn.functional.cosine_similarity(
            corrupted, clean, dim=-1
        ).mean()

    def concentrations(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_concentration) + 0.1

    def _encode_programs(
        self, x: torch.Tensor, need_weights: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        gene_ids = torch.arange(x.shape[1], device=x.device)
        gene_tokens = self.gene_embedding(gene_ids).unsqueeze(0)
        gene_tokens = gene_tokens + self.expression_projection(x.unsqueeze(-1))
        gene_tokens = self.dropout(gene_tokens)
        missing = x <= 0
        if missing.all(dim=1).any():
            missing = missing.clone()
            missing[missing.all(dim=1), 0] = False

        queries = self.program_queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        programs, attention = self.program_attention(
            queries,
            gene_tokens,
            gene_tokens,
            key_padding_mask=missing,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        programs = self.program_norm(queries + self.dropout(programs))
        return programs, attention

    def _scores_from_programs(
        self, programs: torch.Tensor, isoform_queries: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isoform_queries is None:
            isoform_queries = self.isoform_queries()
        compatibility = torch.einsum("td,bmd->btm", isoform_queries, programs)
        compatibility = compatibility / math.sqrt(self.d_model)
        program_weights = compatibility.softmax(dim=-1)
        isoform_context = torch.einsum("btm,bmd->btd", program_weights, programs)
        scores = (isoform_context * isoform_queries.unsqueeze(0)).sum(dim=-1)
        scores = scores * self.score_scale / math.sqrt(self.d_model)
        return scores, program_weights

    def _allocate_from_programs(
        self, programs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores, program_weights = self._scores_from_programs(programs)
        probabilities = grouped_softmax(
            scores, self.isoform_gene_index, self.n_target_genes
        )
        return probabilities, program_weights

    def forward(
        self, x: torch.Tensor, return_concentration: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        programs, _ = self._encode_programs(x, need_weights=False)
        probabilities, _ = self._allocate_from_programs(programs)
        if return_concentration:
            return probabilities, self.concentrations()
        return probabilities

    def attention_maps(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return probabilities, gene-to-program, and program-to-isoform attention."""
        programs, gene_attention = self._encode_programs(x, need_weights=True)
        probabilities, isoform_attention = self._allocate_from_programs(programs)
        if gene_attention is None:
            raise RuntimeError("attention weights were not returned")
        return probabilities, gene_attention, isoform_attention


class HierarchicalIsoBudgetAttentionNet(IsoBudgetAttentionNet):
    """Cluster-backed allocation with a gated within-cluster cell residual.

    A training-domain mean allocation is the explicit prior. A pseudocluster
    context predicts the stable cell-state offset, while a shrinkage gate limits
    the extra cell-specific residual when it does not generalize.
    """

    def __init__(
        self,
        n_genes: int,
        n_isoforms: int,
        isoform_gene_index: torch.Tensor,
        isoform_ids: Sequence[str],
        prior_probabilities: Sequence[float] | torch.Tensor,
        d_model: int = 128,
        n_programs: int = 16,
        n_heads: int = 4,
        dropout: float = 0.10,
        residual_gate_init: float = -1.5,
    ) -> None:
        super().__init__(
            n_genes=n_genes,
            n_isoforms=n_isoforms,
            isoform_gene_index=isoform_gene_index,
            isoform_ids=isoform_ids,
            d_model=d_model,
            n_programs=n_programs,
            n_heads=n_heads,
            dropout=dropout,
        )
        prior = torch.as_tensor(prior_probabilities, dtype=torch.float32)
        if prior.shape != (n_isoforms,) or torch.any(prior <= 0):
            raise ValueError("one strictly positive prior probability is required per isoform")
        prior_sums = torch.zeros(self.n_target_genes, dtype=torch.float32)
        prior_sums.scatter_add_(0, isoform_gene_index.long(), prior)
        prior = prior / prior_sums[isoform_gene_index.long()].clamp_min(1e-8)
        self.register_buffer("prior_logits", prior.log())
        self.cluster_scale_raw = nn.Parameter(torch.tensor(0.5413249))
        self.residual_scale_raw = nn.Parameter(torch.tensor(0.5413249))
        self.cell_gene_gate_logits = nn.Parameter(
            torch.full((self.n_target_genes,), residual_gate_init)
        )
        gate_hidden = max(8, d_model // 4)
        self.cell_state_gate = nn.Sequential(
            nn.Linear(d_model, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )

    def cell_residual_gates(self, cell_programs: torch.Tensor) -> torch.Tensor:
        cell_gate = torch.sigmoid(self.cell_state_gate(cell_programs.mean(dim=1)))
        gene_gate = torch.sigmoid(self.cell_gene_gate_logits)
        return cell_gate * gene_gate.unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        cluster_context: torch.Tensor | None = None,
        return_concentration: bool = False,
        return_components: bool = False,
    ):
        if cluster_context is None:
            cluster_context = x
        cluster_programs, _ = self._encode_programs(
            cluster_context, need_weights=False
        )
        cell_programs, _ = self._encode_programs(x, need_weights=False)
        isoform_queries = self.isoform_queries()
        cluster_scores, _ = self._scores_from_programs(
            cluster_programs, isoform_queries
        )
        cell_scores, _ = self._scores_from_programs(cell_programs, isoform_queries)
        cluster_scale = torch.nn.functional.softplus(self.cluster_scale_raw)
        residual_scale = torch.nn.functional.softplus(self.residual_scale_raw)
        cluster_logits = self.prior_logits.unsqueeze(0) + cluster_scale * cluster_scores
        gene_gates = self.cell_residual_gates(cell_programs)
        isoform_gates = gene_gates[:, self.isoform_gene_index]
        logits = cluster_logits + residual_scale * isoform_gates * (
            cell_scores - cluster_scores
        )
        probabilities = grouped_softmax(
            logits, self.isoform_gene_index, self.n_target_genes
        )
        cluster_probabilities = grouped_softmax(
            cluster_logits, self.isoform_gene_index, self.n_target_genes
        )
        if return_components:
            return (
                probabilities,
                cluster_probabilities,
                self.concentrations(),
                gene_gates,
            )
        if return_concentration:
            return probabilities, self.concentrations()
        return probabilities


def multinomial_allocation_loss(probabilities: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    total = counts.sum().clamp_min(1.0)
    return -(counts * probabilities.clamp_min(1e-8).log()).sum() / total


def shrunken_multinomial_allocation_loss(
    probabilities: torch.Tensor,
    counts: torch.Tensor,
    isoform_gene_index: torch.Tensor,
    prior_probabilities: torch.Tensor,
    prior_strength: float = 2.0,
) -> torch.Tensor:
    """Empirical-Bayes target shrinkage for sparse cell-gene observations."""
    probabilities = probabilities.float().clamp_min(1e-8)
    counts = counts.float()
    groups = isoform_gene_index.unsqueeze(0).expand(counts.shape[0], -1)
    n_groups = int(isoform_gene_index.max().item()) + 1
    gene_counts = torch.zeros(
        (counts.shape[0], n_groups), dtype=counts.dtype, device=counts.device
    )
    gene_counts.scatter_add_(1, groups, counts)
    observed = gene_counts.gather(1, groups) > 0
    pseudo_counts = (
        prior_probabilities.unsqueeze(0) * float(prior_strength) * observed
    )
    targets = counts + pseudo_counts
    return -(targets * probabilities.log()).sum() / targets.sum().clamp_min(1.0)


def dirichlet_multinomial_allocation_loss(
    probabilities: torch.Tensor,
    counts: torch.Tensor,
    isoform_gene_index: torch.Tensor,
    concentrations: torch.Tensor,
) -> torch.Tensor:
    """Dirichlet-multinomial NLL per observed long-read molecule.

    The multinomial coefficient is omitted because it is constant with respect
    to model parameters. Empty gene/cell combinations contribute exactly zero.
    """
    probabilities = probabilities.float().clamp_min(1e-7)
    counts = counts.float()
    concentrations = concentrations.float().clamp(0.1, 1e4)
    groups = isoform_gene_index.unsqueeze(0).expand(counts.shape[0], -1)
    gene_counts = torch.zeros(
        (counts.shape[0], concentrations.numel()),
        dtype=counts.dtype,
        device=counts.device,
    )
    gene_counts.scatter_add_(1, groups, counts)
    alpha = probabilities * concentrations[isoform_gene_index].unsqueeze(0)
    isoform_terms = torch.lgamma(counts + alpha) - torch.lgamma(alpha)
    gene_terms = torch.zeros_like(gene_counts)
    gene_terms.scatter_add_(1, groups, isoform_terms)
    gene_terms += torch.lgamma(concentrations).unsqueeze(0)
    gene_terms -= torch.lgamma(gene_counts + concentrations.unsqueeze(0))
    return -gene_terms.sum() / counts.sum().clamp_min(1.0)
