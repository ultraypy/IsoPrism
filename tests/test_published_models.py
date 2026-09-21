import numpy as np
import pytest
import torch

from short2long.published_models import (METHODS, TabularAdapter, allocation_nll,
                                        build_published_model, chromosome_layout)


GROUPS = np.array([0, 0, 1, 1, 1, 2, 2])
STRUCTURES = ["g0|chr2|+|1-3", "g0|chr2|+|2-3", "g1|chr1|+|1-3", "g1|chr1|+|2-3",
              "g1|chr1|+|4-5", "g2|chr3|-|2-6", "g2|chr3|-|3-6"]


@pytest.mark.parametrize("method", METHODS)
def test_adapted_shapes_probabilities_and_training(method):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    x = torch.rand(8, 12)
    raw = torch.poisson(x * 3)
    y = torch.poisson(torch.rand(8, len(GROUPS)) * 2)
    model = build_published_model(method, 12, GROUPS, STRUCTURES)
    model.train()
    loss, details = model.loss(x, y, raw)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    model.eval()
    p1, p2 = model(x), model(x)
    assert p1.shape == y.shape
    torch.testing.assert_close(p1, p2)
    assert torch.isfinite(p1).all() and (p1 >= 0).all()
    for group in np.unique(GROUPS):
        torch.testing.assert_close(p1[:, GROUPS == group].sum(1), torch.ones(8))
    # Unobserved parent genes carry no supervised composition label.
    assert float(allocation_nll(p1, torch.zeros_like(y)).detach()) == 0


def test_fast_attention_matches_original_forward_and_gradients():
    torch.manual_seed(7)
    reference = TabularAdapter("FT-Transformer", 12, GROUPS, efficient_attention=False).eval()
    fast = TabularAdapter("FT-Transformer", 12, GROUPS).eval()
    fast.load_state_dict(reference.state_dict())
    x1 = torch.rand(4, 12, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    r, f = reference(x1), fast(x2)
    torch.testing.assert_close(r, f, atol=1e-6, rtol=1e-5)
    r[:, 0].sum().backward()
    f[:, 0].sum().backward()
    torch.testing.assert_close(x1.grad, x2.grad, atol=1e-6, rtol=1e-4)


def test_chromosome_permutation_roundtrip():
    order, inverse, sizes, names = chromosome_layout(STRUCTURES)
    np.testing.assert_array_equal(np.arange(7)[order][inverse], np.arange(7))
    assert sum(sizes) == 7 and names == ["chr1", "chr2", "chr3"]


def test_butterfly_pretraining_and_adversarial_gradients():
    model = build_published_model("scButterfly", 12, GROUPS, STRUCTURES).train()
    x, y = torch.rand(8, 12), torch.ones(8, 7)
    for side in ("x", "y"):
        model.zero_grad(set_to_none=True)
        loss = model.pretrain_loss(x, y, side, 0.01)
        loss.backward()
        assert torch.isfinite(loss)
    model.zero_grad(set_to_none=True)
    dloss = model.discriminator_loss(x, y)
    dloss.backward()
    assert all(p.grad is None for p in model.encoder_x.parameters())
    assert any(p.grad is not None for p in model.discriminators.parameters())
