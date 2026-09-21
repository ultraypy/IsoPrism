import numpy as np
import pytest
import torch

from short2long.competitive_model import CompetitiveAllocationNet, multinomial_nll_terms, unsmoothed_cluster_targets


GROUPS = np.array([0, 0, 1, 1, 1])
STRUCTURES = ["g0|chr1|+|10-20", "g0|chr1|+|10-30", "g1|chr2|-|40-60",
              "g1|chr2|-|40-80", "g1|chr2|-|50-90"]


def make_model(**kwargs):
    torch.manual_seed(42)
    return CompetitiveAllocationNet(6, GROUPS, STRUCTURES, d_model=16, n_programs=4, dropout=0, **kwargs)


def test_all_iterations_preserve_gene_simplex_and_uniform_start():
    model = make_model().eval()
    p, states = model(torch.rand(3, 6), return_trajectory=True)
    assert len(states) == 5
    for state in states:
        assert torch.isfinite(state).all() and (state > 0).all()
        torch.testing.assert_close(state[:, :2].sum(1), torch.ones(3))
        torch.testing.assert_close(state[:, 2:].sum(1), torch.ones(3))
    torch.testing.assert_close(states[0][0], torch.tensor([.5, .5, 1/3, 1/3, 1/3]))
    assert not any("prior" in k or "calibration" in k or "reference" in k for k in model.state_dict())


def test_state_changes_rates_and_frozen_iterations_equal_one_step():
    model = make_model().eval()
    x = torch.rand(3, 6)
    one = model(x, steps=1)
    frozen = model(x, steps=4, frozen_state=True)
    torch.testing.assert_close(one, frozen, atol=1e-6, rtol=1e-6)
    iterative = model(x)
    assert float((iterative - frozen).abs().max().detach()) > 1e-6


def test_all_zero_expression_is_finite():
    assert torch.isfinite(make_model()(torch.zeros(2, 6))).all()


@pytest.mark.parametrize("mode", ["structure", "identity"])
def test_count_likelihood_backpropagates_through_shared_field(mode):
    model = make_model(query_mode=mode)
    y = torch.tensor([[2, 0, 1, 3, 0], [0, 4, 0, 2, 1]], dtype=torch.float32)
    numerator, denominator = multinomial_nll_terms(model(torch.rand(2, 6)), y)
    (numerator / denominator).backward()
    for parameter in model.state_feedback.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert sum(p.grad.abs().sum().item() for p in model.state_feedback.parameters()) > 0


def test_count_nll_has_no_pseudocounts_and_zero_counts_have_zero_gradient():
    p = torch.tensor([[.3, .7, .2, .3, .5]], requires_grad=True)
    y = torch.tensor([[2., 0., 0., 0., 0.]])
    numerator, denominator = multinomial_nll_terms(p, y)
    torch.testing.assert_close(numerator, -2 * torch.log(p[0, 0]))
    assert denominator == 2
    numerator.backward()
    assert (p.grad[0, 1:] == 0).all()


def test_cluster_supervision_is_equal_cell_and_not_pooled_or_smoothed():
    y = np.array([[100, 0], [0, 2], [1, 0], [0, 0]], dtype=np.float32)
    targets, valid = unsmoothed_cluster_targets(y, np.zeros(4, dtype=int), np.zeros(2, dtype=int), min_cells=2)
    np.testing.assert_allclose(targets, [[.5, .5]])
    assert valid.tolist() == [[True]]
    targets, valid = unsmoothed_cluster_targets(y, np.arange(4), np.zeros(2, dtype=int))
    assert not valid.any()
    np.testing.assert_allclose(targets[-1], [0, 0])


def test_checkpoint_reload_uses_no_long_read_inputs():
    first, second = make_model().eval(), make_model().eval()
    second.load_state_dict(first.state_dict())
    x, context = torch.rand(2, 6), torch.rand(2, 6)
    torch.testing.assert_close(first(x, context), second(x, context))


def test_structure_queries_are_equivariant_to_isoform_reordering():
    first = make_model().eval()
    perm = np.array([1, 0, 4, 2, 3])
    second = CompetitiveAllocationNet(6, GROUPS[perm], [STRUCTURES[i] for i in perm],
                                     d_model=16, n_programs=4, dropout=0).eval()
    with torch.no_grad():
        source = dict(first.named_parameters())
        for name, parameter in second.named_parameters():
            parameter.copy_(source[name])
    x = torch.rand(2, 6)
    torch.testing.assert_close(first(x)[:, perm], second(x), atol=1e-6, rtol=1e-6)


def test_bad_shapes_and_steps_are_rejected():
    with pytest.raises(ValueError):
        make_model(steps=0)
    with pytest.raises(ValueError):
        make_model()(torch.rand(2, 6), torch.rand(2, 5))
