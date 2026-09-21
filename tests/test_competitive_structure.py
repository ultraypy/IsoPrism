import numpy as np
import pytest
import torch

from short2long.competitive_model import CompetitiveAllocationNet
from short2long.competitive_structure import annotation_terminals, representation_collisions


def test_endpoint_variants_share_chain_but_not_terminal_descriptors(tmp_path):
    path = tmp_path / "example.gtf"
    path.write_text('chr1\ts\ttranscript\t1\t60\t.\t-\t.\ttranscript_id "a";\n'
                    'chr1\ts\texon\t1\t20\t.\t-\t.\ttranscript_id "a";\n'
                    'chr1\ts\texon\t40\t60\t.\t-\t.\ttranscript_id "a";\n'
                    'chr1\ts\ttranscript\t6\t60\t.\t-\t.\ttranscript_id "b";\n'
                    'chr1\ts\texon\t6\t20\t.\t-\t.\ttranscript_id "b";\n'
                    'chr1\ts\texon\t40\t60\t.\t-\t.\ttranscript_id "b";\n')
    records = annotation_terminals(str(path), ("a", "b"))
    assert records["a"]["features"][0] == records["b"]["features"][0]
    assert records["a"]["features"][1] != records["b"]["features"][1]
    structure = ["g|chr1|-|20-40"] * 2
    terminal = np.array([records[name]["features"] + [1] for name in ["a", "b"]], np.float32)
    assert representation_collisions([0, 0], structure) == [[0, 1]]
    assert representation_collisions([0, 0], structure, terminal) == []


def test_terminal_aware_model_breaks_forced_equal_allocation_and_reloads():
    torch.manual_seed(19)
    structure = ["g|chr1|+|20-40"] * 2
    kwargs = dict(n_input=3, groups=[0, 0], structures=structure, d_model=8,
                  n_programs=2, n_heads=2, dropout=0)
    x = torch.rand(12, 3)
    old = CompetitiveAllocationNet(**kwargs).eval()
    torch.testing.assert_close(old(x), torch.full((12, 2), .5))
    terminal = [[0, 0, 0, 0, 1], [1, -1, .3, .2, 1]]
    fixed = CompetitiveAllocationNet(**kwargs, terminal_features=terminal).eval()
    p = fixed(x)
    assert float((p[:, 0] - .5).abs().max().detach()) > 1e-5
    assert float(p[:, 0].std().detach()) > 1e-7
    torch.testing.assert_close(p.sum(1), torch.ones(12))
    reloaded = CompetitiveAllocationNet(**kwargs, terminal_features=terminal).eval()
    reloaded.load_state_dict(fixed.state_dict())
    torch.testing.assert_close(p, reloaded(x))
    with pytest.raises(ValueError):
        CompetitiveAllocationNet(**kwargs, terminal_features=[[0], [1]])
