from types import SimpleNamespace
import numpy as np
from scipy import sparse
import torch

from short2long.unified_data import select_catalogue
from short2long.unified_benchmark import PlainMLP, classical_fit, classical_predict
from short2long.unified_metrics import moran, spatial_metrics
from short2long.coverage_retrain import evaluate_sparse


def test_catalogue_does_not_read_heldout_labels():
    d = SimpleNamespace(x=sparse.csr_matrix(np.ones((8,3))),
        y=sparse.csr_matrix([[2,2,0,0],[2,2,0,0],[1,1,0,0],[1,1,0,0],
                             [0,0,5,5],[0,0,5,5],[0,0,5,5],[0,0,5,5]]),
        isoform_genes=np.array(["a","a","b","b"]))
    first = select_catalogue(d,np.arange(4))
    changed = d.y.toarray(); changed[4:] = 10000
    d.y = sparse.csr_matrix(changed)
    second = select_catalogue(d,np.arange(4))
    np.testing.assert_array_equal(first[0],second[0]); np.testing.assert_array_equal(first[1],second[1])
    np.testing.assert_array_equal(first[1],[0,1])


def test_allocation_mlp_normalizes_all_genes():
    model = PlainMLP(3,np.array([0,0,1,1,1]))
    p = model(torch.randn(7,3))
    torch.testing.assert_close(p[:,:2].sum(1),torch.ones(7))
    torch.testing.assert_close(p[:,2:].sum(1),torch.ones(7))
    assert (p >= 0).all()


def example_view():
    rng = np.random.default_rng(5)
    return {"fit_x":rng.normal(size=(30,4)).astype(np.float32),
            "fit_y":rng.poisson(3,size=(30,4)).astype(np.float32),"groups":np.array([0,0,1,1])}


def test_classical_proportions_are_valid():
    view = example_view()
    for method,params in [("Mean",{}),("PCA-kNN",{"k":15,"prior_strength":1}),
                           ("PCA-Ridge",{"ridge_alpha":10})]:
        model = classical_fit(method,view,"fit",params)
        p = classical_predict(model,view["fit_x"][:5])
        assert np.isfinite(p).all() and (p >= 0).all()
        np.testing.assert_allclose(p[:,:2].sum(1),1,atol=1e-6)
        np.testing.assert_allclose(p[:,2:].sum(1),1,atol=1e-6)


def test_mean_is_pooled_count_prior_not_cell_mean():
    view = {"fit_x":np.ones((2,2)),"fit_y":np.array([[100,0],[0,2]]),"groups":np.array([0,0])}
    model = classical_fit("Mean",view,"fit",{})
    np.testing.assert_allclose(model["prior"],[100.5/103,2.5/103],rtol=1e-6)


def test_moran_constant_is_undefined():
    assert np.isnan(moran(np.ones(30),np.tile(np.arange(6),(30,1))))


def test_spatial_panel_and_missingness_are_explicit():
    coords = np.column_stack((np.arange(30),np.zeros(30)))
    rng = np.random.default_rng(7)
    y = rng.poisson(10,size=(30,4)).astype(np.float32)
    groups = np.array([0,0,1,1])
    same = y.copy(); same[:,:2] /= y[:,:2].sum(1,keepdims=True); same[:,2:] /= y[:,2:].sum(1,keepdims=True)
    result = spatial_metrics(y,same,groups,coords,np.arange(4))
    assert np.isclose(result["spatial_proportion_moran_spearman"],1)
    const = spatial_metrics(y,np.full_like(y,.5),groups,coords,np.arange(4))
    assert const["spatial_proportion_moran_spearman"] is None
    assert const["spatial_prediction_valid_size"] == 0
    assert const["spatial_truth_panel_size"] == result["spatial_truth_panel_size"] == 4


def test_metric_eligibility_does_not_depend_on_prediction():
    y = sparse.csr_matrix([[2,0,1,0],[1,1,0,0],[0,0,2,2],[0,4,0,2]])
    groups = np.array([0,0,1,1]); labels = np.array([0,0,1,1])
    left = evaluate_sparse(y,np.full((4,4),.5,dtype=np.float32),groups,labels)
    right = evaluate_sparse(y,np.tile([.9,.1,.1,.9],(4,1)).astype(np.float32),groups,labels)
    for key in ["n_evaluable_cell_genes","n_confident_dominant_pairs","n_genes_confident_dominant"]:
        assert left[key] == right[key]
