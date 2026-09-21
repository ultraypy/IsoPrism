"""Task adapters for published networks, not claims of unmodified reproduction.

Upstream implementations live in pinned, unmodified ``third_party`` checkouts.
Only the prediction contract and modality-specific likelihoods are adapted.
"""
from __future__ import annotations

import ast
import importlib.util
import logging
import sys
import types
import typing
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import _parse_splice_chain, grouped_softmax

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = {
    "rtdl": "e3ed46cac38568785289d8fa16b8cfa585bde27e",
    "babel": "55742e4c704eb0e1e81569c7caeceb0a516bd90b",
    "scButterfly": "eb31e04bb8c4abdf85c4cbecd044fcd359105caa",
}
METHODS = ("TabResNet", "FT-Transformer", "BABEL", "scButterfly")


@lru_cache(None)
def load_source(relative: str):
    path = ROOT / "third_party" / relative
    spec = importlib.util.spec_from_file_location(
        "short2long_upstream_" + relative.replace("/", "_").replace(".", "_"), path
    )
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(None)
def babel_components():
    """Load original neural classes without importing the optional skorch trainer.

    Class bodies are compiled verbatim from the pinned source AST. No definitions,
    activations, or weights are replaced with stand-ins for missing dependencies.
    """
    path = ROOT / "third_party/babel/babel/models/autoencoders.py"
    names = {"Encoder", "Decoder", "ChromEncoder", "ChromDecoder"}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError("the pinned BABEL class definitions have changed")
    namespace = {
        "__name__": "short2long_babel_components", "torch": torch, "nn": nn,
        "F": F, "logging": logging, "List": typing.List,
        "activations": load_source("babel/babel/activations.py"),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return types.SimpleNamespace(**{name: namespace[name] for name in names})


def sdpa_forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
    """Exact scaled-dot-product attention using PyTorch's memory-efficient kernel."""
    if self.key_compression is not None:
        raise ValueError("this adapter intentionally does not compress feature tokens")
    b, nq, d = x_q.shape
    h = self._n_heads
    q = self.W_q(x_q).view(b, nq, h, d // h).transpose(1, 2)
    k = self.W_k(x_kv).view(b, -1, h, d // h).transpose(1, 2)
    v = self.W_v(x_kv).view(b, -1, h, d // h).transpose(1, 2)
    dropout = self.dropout.p if self.training and self.dropout is not None else 0.0
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)
    out = out.transpose(1, 2).reshape(b, nq, d)
    return out if self.W_out is None else self.W_out(out)


def chromosome_layout(structures: list[str]):
    # A fixed annotation is permitted; no held-out long-read counts are consulted.
    chroms = [_parse_splice_chain(s)[0] for s in structures]
    order = np.asarray(sorted(range(len(chroms)), key=lambda i: (chroms[i], i)), dtype=np.int64)
    names, sizes = np.unique(np.asarray(chroms)[order], return_counts=True)
    return order, np.argsort(order), sizes.tolist(), names.tolist()


def allocation_nll(probabilities: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    return -(counts.float() * probabilities.float().clamp_min(1e-8).log()).sum() / counts.sum().clamp_min(1)


def nb_loss(mu: torch.Tensor, theta: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """RNA auxiliary likelihood, conditioned only on the observed short-read library."""
    y = counts.float()
    mu = mu.float().clamp(1e-5, 1e6) * (y.sum(1, keepdim=True).clamp_min(1) / 1e4)
    theta = theta.float().clamp(1e-4, 1e3)
    logsum = (mu + theta).log()
    logp = (torch.lgamma(y + theta) - torch.lgamma(theta) - torch.lgamma(y + 1)
            + theta * (theta.log() - logsum) + y * (mu.log() - logsum))
    return -logp.mean()


class AllocationAdapter(nn.Module):
    def __init__(self, group_index):
        super().__init__()
        self.register_buffer("group_index", torch.as_tensor(group_index, dtype=torch.long))
        self.n_groups = int(self.group_index.max()) + 1

    def proportions(self, logits):
        return grouped_softmax(logits.float(), self.group_index, self.n_groups)

    def observed_fractions(self, counts):
        totals = counts.new_zeros((len(counts), self.n_groups))
        groups = self.group_index[None, :].expand_as(counts)
        totals.scatter_add_(1, groups, counts)
        # A wholly unobserved gene has an all-zero input block, not a uniform label.
        return counts / totals.gather(1, groups).clamp_min(1)

    def loss(self, x, y, raw_x=None, **kwargs):
        p = self(x)
        loss = allocation_nll(p, y)
        return loss, {"allocation": loss.detach()}


class TabularAdapter(AllocationAdapter):
    def __init__(self, method, n_input, group_index, efficient_attention=True):
        super().__init__(group_index)
        rtdl = load_source("rtdl/package/rtdl_revisiting_models.py")
        self.method = method
        if method == "TabResNet":
            self.network = rtdl.ResNet(
                d_in=n_input, d_out=len(group_index), n_blocks=3, d_block=256,
                d_hidden_multiplier=2.0, dropout1=0.15, dropout2=0.0,
            )
        elif method == "FT-Transformer":
            self.network = rtdl.FTTransformer(
                n_cont_features=n_input, cat_cardinalities=[], d_out=len(group_index),
                **rtdl.FTTransformer.get_default_kwargs(n_blocks=2),
            )
            if efficient_attention:
                for block in self.network.backbone.blocks:
                    attention = block["attention"]
                    attention.forward = types.MethodType(sdpa_forward, attention)
        else:
            raise ValueError(method)

    def forward(self, x):
        logits = self.network(x, None) if self.method == "FT-Transformer" else self.network(x)
        return self.proportions(logits)


class BabelAdapter(AllocationAdapter):
    """BABEL four-path training with the original chromosome-aware neural blocks."""
    def __init__(self, n_input, group_index, structures):
        super().__init__(group_index)
        babel = babel_components()
        activation = load_source("babel/babel/activations.py")
        order, inverse, sizes, _ = chromosome_layout(structures)
        self.register_buffer("order", torch.as_tensor(order))
        self.register_buffer("inverse", torch.as_tensor(inverse))
        self.chrom_sizes = sizes
        self.encoder_x = babel.Encoder(n_input, num_units=16)
        self.encoder_y = babel.ChromEncoder(sizes, latent_dim=16)
        self.decoder_x = babel.Decoder(
            n_input, num_units=16,
            final_activation=[activation.Exp(), activation.ClippedSoftplus()],
        )
        self.decoder_y = babel.ChromDecoder(sizes, latent_dim=16, final_activations=None)
        # Extra upstream NB/zero-inflation heads remain structurally intact but
        # their outputs are unused by the isoform composition likelihood.

    def encode_y(self, y):
        fractions = self.observed_fractions(y)[:, self.order]
        return self.encoder_y(torch.split(fractions, self.chrom_sizes, dim=1))

    def decode_y(self, z):
        return self.proportions(self.decoder_y(z)[0][:, self.inverse])

    def forward(self, x):
        return self.decode_y(self.encoder_x(x))

    def loss(self, x, y, raw_x=None, **kwargs):
        if raw_x is None:
            raise ValueError("BABEL RNA reconstruction requires short-read counts")
        zx, zy = self.encoder_x(x), self.encode_y(y)
        xy, yy = self.decode_y(zx), self.decode_y(zy)
        xx, yx = self.decoder_x(zx), self.decoder_x(zy)
        supervised = allocation_nll(xy, y)
        iso_reconstruction = allocation_nll(yy, y)
        rna = nb_loss(xx[0], xx[1], raw_x) + nb_loss(yx[0], yx[1], raw_x)
        loss = rna + 3.0 * (supervised + iso_reconstruction)
        return loss, {"allocation": supervised.detach(), "iso_reconstruction": iso_reconstruction.detach(), "rna_nb": rna.detach()}


class ButterflyAdapter(AllocationAdapter):
    """scButterfly-B adapter using the original NetBlock and Translator classes.

    Preserve masked modality VAEs, four decoded paths and dual discriminators.
    No label-based re-pairing, external pretraining or target-domain adaptation.
    """
    def __init__(self, n_input, group_index, structures):
        super().__init__(group_index)
        scb = load_source("scButterfly/scButterfly/model_component.py")
        order, inverse, sizes, _ = chromosome_layout(structures)
        self.register_buffer("order", torch.as_tensor(order))
        self.register_buffer("inverse", torch.as_tensor(inverse))
        hidden_y = 32 * len(sizes)
        self.encoder_x = scb.NetBlock(2, [n_input, 256, 128], [nn.LeakyReLU(), nn.LeakyReLU()], 0.1, 0.5)
        self.encoder_y = scb.Split_Chrom_Encoder_block(
            2, [len(group_index), hidden_y, 128], [nn.LeakyReLU(), nn.LeakyReLU()], sizes, 0.1, 0.3,
        )
        self.decoder_x = scb.NetBlock(2, [128, 256, n_input], [nn.LeakyReLU(), nn.LeakyReLU()], 0.1, 0)
        self.decoder_y = scb.Split_Chrom_Decoder_block(
            2, [128, hidden_y, len(group_index)], [nn.LeakyReLU(), nn.Identity()], sizes, 0.1, 0,
        )
        self.translator = scb.Translator(128, 128, 128, [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()])
        # The diagonal paths of the joint translator are exactly the single-VAE
        # paths, so pretraining them in place avoids unrelated random reinitialization.
        self.discriminators = nn.ModuleDict({
            side: scb.NetBlock(1, [128, 1], [nn.Sigmoid()], 0, 0) for side in ("x", "y")
        })

    def encode_y(self, y):
        return self.encoder_y(self.observed_fractions(y)[:, self.order])

    def decode_y(self, z):
        return self.proportions(self.decoder_y(z)[:, self.inverse])

    def translate(self, z, side, deterministic=False):
        mode = "test" if deterministic or not self.training else "train"
        if side == "x":
            return self.translator.forward_with_RNA(z, mode)
        return self.translator.forward_with_ATAC(z, mode)

    def forward(self, x):
        _, xy, _, _ = self.translate(self.encoder_x(x), "x", deterministic=True)
        return self.decode_y(xy)

    @staticmethod
    def kl(mu, logvar):
        return -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()

    def pretrain_loss(self, x, y, side, kl_weight):
        if side == "x":
            xx, _, mu, lv = self.translate(self.encoder_x(x), "x")
            recon = F.mse_loss(self.decoder_x(xx), x)
        else:
            _, yy, mu, lv = self.translate(self.encode_y(y), "y")
            recon = allocation_nll(self.decode_y(yy), y)
        return recon + kl_weight * self.kl(mu, lv)

    def latent_paths(self, x, y):
        zx, zy = self.encoder_x(x), self.encode_y(y)
        xx, xy, mx, lx = self.translate(zx, "x")
        yx, yy, my, ly = self.translate(zy, "y")
        return zx, zy, xx, xy, yx, yy, mx, lx, my, ly

    def adversarial_loss(self, zx, zy, xy, yx, detach):
        # Same soft-label discriminator game as the upstream training routine.
        label = torch.rand(len(zx), device=zx.device)
        label = torch.where((label > 0.5) & (label <= 0.8), 0.8, label)
        label = torch.where((label > 0.2) & (label <= 0.5), 0.2, label)
        mask = (label > 0.5)[:, None]
        mix_x = torch.where(mask, zx, yx)
        mix_y = torch.where(mask, zy, xy)
        if detach:
            mix_x, mix_y = mix_x.detach(), mix_y.detach()
        dx = self.discriminators["x"](mix_x).flatten().clamp(1e-6, 1 - 1e-6)
        dy = self.discriminators["y"](mix_y).flatten().clamp(1e-6, 1 - 1e-6)
        return F.binary_cross_entropy(dx, label) + F.binary_cross_entropy(dy, label)

    def discriminator_loss(self, x, y):
        with torch.no_grad():
            zx, zy, _, xy, yx, *_ = self.latent_paths(x, y)
        return self.adversarial_loss(zx, zy, xy, yx, detach=True)

    def loss(self, x, y, raw_x=None, kl_weight=0.01, **kwargs):
        zx, zy, xx, xy, yx, yy, mx, lx, my, ly = self.latent_paths(x, y)
        supervised = allocation_nll(self.decode_y(xy), y)
        iso_reconstruction = allocation_nll(self.decode_y(yy), y)
        rna = F.mse_loss(self.decoder_x(xx), x) + F.mse_loss(self.decoder_x(yx), x)
        kl = self.kl(mx, lx) + self.kl(my, ly)
        adversarial = self.adversarial_loss(zx, zy, xy, yx, detach=False)
        loss = rna + 3.0 * (supervised + iso_reconstruction) + kl_weight * kl
        # Retain the discriminator-confidence threshold in upstream scButterfly.
        loss = loss - 0.1 * adversarial * (adversarial.detach() < 1.35)
        return loss, {"allocation": supervised.detach(), "iso_reconstruction": iso_reconstruction.detach(),
                      "rna_mse": rna.detach(), "kl": kl.detach(), "adversarial": adversarial.detach()}


def build_published_model(method, n_input, group_index, structures):
    if method in ("TabResNet", "FT-Transformer"):
        return TabularAdapter(method, n_input, group_index)
    if method == "BABEL":
        return BabelAdapter(n_input, group_index, structures)
    if method == "scButterfly":
        return ButterflyAdapter(n_input, group_index, structures)
    raise ValueError(method)
