from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

Array = np.ndarray


@dataclass
class IsoVAEConfig:
    """Neural network configuration for IsoVAE."""

    gene_hidden: Tuple[int, int] = (512, 256)
    iso_hidden: Tuple[int, int] = (512, 256)
    decoder_hidden: Tuple[int, int] = (256, 512)
    latent_dim: int = 32
    dropout: float = 0.20


class GaussianEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: Tuple[int, int],
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(h2, latent_dim)
        self.logvar = nn.Linear(h2, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), torch.clamp(self.logvar(h), min=-10.0, max=6.0)


class IsoVAEModel(nn.Module):
    """Multimodal hierarchical VAE for isoform-usage prediction and denoising."""

    def __init__(
        self,
        n_gene_inputs: int,
        n_isoforms: int,
        n_gene_groups: int,
        config: IsoVAEConfig,
    ) -> None:
        super().__init__()
        self.n_gene_inputs = int(n_gene_inputs)
        self.n_isoforms = int(n_isoforms)
        self.n_gene_groups = int(n_gene_groups)
        self.latent_dim = int(config.latent_dim)

        self.gene_encoder = GaussianEncoder(
            self.n_gene_inputs, config.gene_hidden, config.latent_dim, config.dropout
        )
        self.iso_encoder = GaussianEncoder(
            self.n_isoforms + self.n_gene_groups,
            config.iso_hidden,
            config.latent_dim,
            config.dropout,
        )
        d1, d2 = config.decoder_hidden
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, d1),
            nn.LayerNorm(d1),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d1, d2),
            nn.LayerNorm(d2),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.iso_head = nn.Linear(d2, self.n_isoforms)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode_gene(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.gene_encoder(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return z, mu, logvar

    def encode_iso(self, iso_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.iso_encoder(iso_input)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return z, mu, logvar

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.iso_head(self.decoder(z))

    def predict_from_gene(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encode_gene(x)
        return self.decode_logits(z), mu, logvar

    def denoise_from_iso(self, iso_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encode_iso(iso_input)
        return self.decode_logits(z), mu, logvar


def usage_from_counts_torch(
    y_counts: torch.Tensor,
    group_tensors: Sequence[torch.Tensor],
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert count tensor to usage tensor plus gene-level coverage features."""
    usage = torch.zeros_like(y_counts)
    covs: List[torch.Tensor] = []
    for idx in group_tensors:
        counts_g = y_counts.index_select(1, idx)
        cov = counts_g.sum(dim=1, keepdim=True)
        usage_g = counts_g / (cov + eps)
        usage[:, idx] = torch.where(cov > 0, usage_g, torch.zeros_like(usage_g))
        covs.append(cov)
    return usage, torch.cat(covs, dim=1)


def iso_encoder_input_from_counts(
    y_counts: torch.Tensor,
    group_tensors: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Build long-read encoder input: isoform usage + standardized log gene coverage."""
    usage, cov = usage_from_counts_torch(y_counts, group_tensors)
    cov_feat = torch.log1p(cov)
    cov_feat = (cov_feat - cov_feat.mean(dim=1, keepdim=True)) / (
        cov_feat.std(dim=1, keepdim=True) + 1e-6
    )
    return torch.cat([usage, cov_feat], dim=1)


def logits_to_usage(
    logits: torch.Tensor,
    groups: Sequence[Array],
) -> Array:
    """Apply gene-wise softmax to logits and return isoform-usage proportions."""
    out = np.zeros(tuple(logits.shape), dtype=np.float32)
    for idx_np in groups:
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=logits.device)
        out[:, idx_np] = torch.softmax(logits.index_select(1, idx), dim=1).detach().cpu().numpy()
    return out


def make_config_from_checkpoint(config_dict: dict, state_dict: Optional[dict] = None) -> IsoVAEConfig:
    """Create a clean IsoVAEConfig from older checkpoints.

    Historical checkpoints may contain extra keys from attention experiments.
    They are ignored here so that the public package loads the final model.
    """
    allowed = {"gene_hidden", "iso_hidden", "decoder_hidden", "latent_dim", "dropout"}
    clean = {k: v for k, v in (config_dict or {}).items() if k in allowed}
    if "gene_hidden" in clean:
        clean["gene_hidden"] = tuple(clean["gene_hidden"])
    if "iso_hidden" in clean:
        clean["iso_hidden"] = tuple(clean["iso_hidden"])
    if "decoder_hidden" in clean:
        clean["decoder_hidden"] = tuple(clean["decoder_hidden"])
    return IsoVAEConfig(**clean)
