import json
import numpy as np
import pandas as pd
import pytest
import torch

from isoprism import IsoPrism
from isoprism.benchmark import METHODS, build_queue, main, verify_prepared
from isoprism.report import export_tables
from short2long.competitive_model import CompetitiveAllocationNet
from short2long.published_benchmark import file_hash


def test_legacy_model_alias_preserves_state():
    assert IsoPrism is CompetitiveAllocationNet
    kwargs = dict(n_input=3, groups=[0, 0], structures=['g|chr1|+|10-20', 'g|chr1|+|10-30'],
                  d_model=8, n_programs=4, n_heads=2, dropout=0., terminal_features=np.zeros((2, 5)))
    old, new = CompetitiveAllocationNet(**kwargs).eval(), IsoPrism(**kwargs).eval()
    new.load_state_dict(old.state_dict())
    x = torch.ones(2, 3)
    torch.testing.assert_close(old(x), new(x), rtol=0, atol=0)


def test_exact_table_method_list_and_queue():
    assert METHODS == ['PCA-kNN', 'MLP', 'Context-MLP', 'TabResNet', 'FT-Transformer', 'BABEL', 'scButterfly', 'IsoPrism']
    queue = build_queue(['a', 'b', 'c', 'd', 'e'], METHODS, [2026, 2027, 2028])
    assert len(queue) == len(set(queue)) == 110
    assert len([q for q in queue if q[1] == 'PCA-kNN']) == 5


def test_dry_run_requires_neither_data_nor_cuda(tmp_path, capsys):
    main(['--prepared-root', str(tmp_path / 'absent'), '--output-root', str(tmp_path / 'out'),
          '--methods', 'IsoPrism', '--dry-run'])
    assert json.loads(capsys.readouterr().out)['planned_runs'] == 15
    assert not (tmp_path / 'out').exists()


def test_frozen_manifest_is_checked(tmp_path):
    folder = tmp_path / 'prepared/cross_donor'
    folder.mkdir(parents=True)
    data = folder / 'example.txt'
    data.write_text('original')
    path = folder / 'manifest.json'
    record = dict(task='cross_donor', smoke_cells=None, source_ids=['s'], test_ids=['t'],
                  prepared_hashes={'example.txt': file_hash(data)})
    path.write_text(json.dumps(record))
    assert verify_prepared(tmp_path, ['cross_donor'])['cross_donor'] == file_hash(path)
    data.write_text('changed')
    with pytest.raises(ValueError, match='Changed preparation'):
        verify_prepared(tmp_path, ['cross_donor'])


def test_report_keeps_max_median_points_and_missingness(tmp_path):
    for seed, values in [(2026, [.2, .8]), (2027, [.4, .6])]:
        folder = tmp_path / f'cross_donor/IsoPrism/seed_{seed}'
        folder.mkdir(parents=True)
        metrics = dict(confident_dominant_isoform_gene_macro_accuracy=.9, cell_gene_1_minus_jsd=.8,
                       cluster_allocation_spearman_values=values, cluster_allocation_spearman_cluster_ids=[0, 1],
                       cluster_allocation_spearman_median=.5, spatial_proportion_moran_spearman=None)
        (folder / 'result.json').write_text(json.dumps(dict(task='cross_donor', variant='IsoPrism', seed=seed, metrics=metrics)))
    export_tables(tmp_path)
    runs = pd.read_csv(tmp_path / 'individual_runs.csv')
    np.testing.assert_allclose(runs.cluster_spearman_max, [.8, .6])
    np.testing.assert_allclose(runs.cluster_spearman_median, [.5, .5])
    assert runs.spatial_moran_spearman.isna().all()
    assert len(pd.read_csv(tmp_path / 'cluster_points.csv')) == 4
    summary = pd.read_csv(tmp_path / 'summary.csv').set_index('metric')
    assert summary.loc['cluster_spearman_max', 'mean'] == pytest.approx(.7)
    assert summary.loc['cluster_spearman_max', 'sample_sd'] == pytest.approx(np.std([.8, .6], ddof=1))
    assert summary.loc['spatial_moran_spearman', 'defined_runs'] == 0
