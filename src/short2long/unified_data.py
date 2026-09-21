"""Frozen, nested source-only preparation for the 2026-09 unified benchmark."""
from pathlib import Path
import json

import numpy as np
from scipy import sparse

from .conservative_train import make_context, source_split
from .data import load_dataset, choose_input_genes, normalize_log_cpm
from .matrix_benchmark import fit_pca, short_read_evaluation_groups
from .published_benchmark import file_hash, write_json
from .train import filter_targets, retain_multi_isoform_genes, split_indices

SPECS = {
    "cross_donor": ("data/processed/crc_paired", "donor", "PS018"),
    "cross_dataset": ("data/processed/harmonized_ont_pacbio_author_v3", "domain", "FLAMES_ONT"),
    "cross_disease": ("data/processed/crc_paired", "condition", "tumor"),
    "cross_platform": ("data/processed/harmonized_ont_pacbio_author_v3", "domain", "CRC_PacBio"),
    "spatial": ("data/processed/post_mi_spatial", "section", "D"),
}
SPLIT_SEED = 2026


def select_catalogue(dataset, rows):
    gi = choose_input_genes(dataset.x[rows], 2000)
    yi = retain_multi_isoform_genes(filter_targets(dataset.y, rows, 3, 2), dataset.isoform_genes)
    if not len(yi):
        raise ValueError("No source-supported multi-isoform genes")
    return gi, yi


def describe_catalogue(dataset, gi, yi):
    parents, groups = np.unique(dataset.isoform_genes[yi], return_inverse=True)
    return {"input_genes": dataset.genes[gi].tolist(), "isoforms": dataset.isoforms[yi].tolist(),
            "parents": parents.tolist(), "groups": groups.tolist(),
            "structures": (dataset.isoform_structures[yi] if dataset.isoform_structures is not None
                           else dataset.isoforms[yi]).tolist()}


def save_partition(folder, name, dataset, rows, gi, yi, with_context=True):
    raw = dataset.x[rows][:, gi].toarray().astype(np.float32)
    x = normalize_log_cpm(dataset.x[rows][:, gi])
    np.save(folder/f"{name}_x.npy", x)
    np.save(folder/f"{name}_raw.npy", raw)
    # Keep held-out truth physically separate from training arrays.
    y = dataset.y[rows][:, yi].tocsr().astype(np.float32)
    if name == "test":
        sparse.save_npz(folder/"test_truth.npz", y)
    else:
        np.save(folder/f"{name}_y.npy", y.toarray())
    np.save(folder/f"{name}_ids.npy", dataset.obs.index[rows].to_numpy(dtype=str))
    if with_context:
        context, means, labels = make_context(x, SPLIT_SEED)
        np.save(folder/f"{name}_context.npy", context)
        np.save(folder/f"{name}_context_means.npy", means)
        np.save(folder/f"{name}_context_labels.npy", labels)


def prepare(task, root, smoke_cells=None):
    folder = Path(root)/"prepared"/task
    if (folder/"manifest.json").exists():
        manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
        if manifest["smoke_cells"] != smoke_cells:
            raise ValueError("Do not mix smoke and full preparations")
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    path, column, holdout = SPECS[task]
    dataset = load_dataset(Path(path))
    source, test, _ = split_indices(dataset.obs, column, holdout, SPLIT_SEED)
    fit, val, cal, strategy = source_split(dataset.obs.iloc[source], SPLIT_SEED)
    parts = {"fit": source[fit], "val": source[val], "cal": source[cal]}
    if smoke_cells:
        rng = np.random.default_rng(SPLIT_SEED)
        # Catalogue remains full-size to exercise the real output-layer memory.
        sampled = {k: np.sort(rng.choice(v, min(len(v), smoke_cells), replace=False)) for k,v in parts.items()}
        test_run = np.sort(rng.choice(test, min(len(test), 64), replace=False))
    else:
        sampled, test_run = parts, test
    inner_gi, inner_yi = select_catalogue(dataset, parts["fit"])
    final_gi, final_yi = select_catalogue(dataset, source)
    inner, final = folder/"inner", folder/"final"
    inner.mkdir(exist_ok=True); final.mkdir(exist_ok=True)
    for name, rows in sampled.items():
        save_partition(inner, name, dataset, rows, inner_gi, inner_yi)
    final_train = np.sort(np.concatenate(list(sampled.values())))
    save_partition(final, "train", dataset, final_train, final_gi, final_yi)
    save_partition(final, "test", dataset, test_run, final_gi, final_yi)
    for target, gi, yi in [(inner, inner_gi, inner_yi), (final, final_gi, final_yi)]:
        write_json(target/"catalogue.json", describe_catalogue(dataset, gi, yi))
    sx = np.load(final/"train_x.npy", mmap_mode="r")
    tx = np.load(final/"test_x.npy", mmap_mode="r")
    _, te, _ = fit_pca(sx, tx, 32, SPLIT_SEED)
    labels, _, grouping = short_read_evaluation_groups(te, SPLIT_SEED)
    np.save(final/"evaluation_clusters.npy", labels)
    # The spatial candidate panel is selected on SOURCE counts, before testing.
    bulk = np.asarray(dataset.y[source][:,final_yi].sum(0)).ravel()
    panel = np.argsort(-bulk, kind="stable")[:250]
    np.save(final/"spatial_panel.npy", panel)
    if task == "spatial":
        np.save(final/"coordinates.npy", dataset.obs.iloc[test_run][["spatial_x", "spatial_y"]].to_numpy(float))
    src_donors = set(dataset.obs.iloc[source]["donor"].dropna().astype(str)) if "donor" in dataset.obs else set()
    test_donors = set(dataset.obs.iloc[test]["donor"].dropna().astype(str)) if "donor" in dataset.obs else set()
    manifest = {"task": task, "data": path, "holdout": {column: holdout}, "split_seed": SPLIT_SEED,
        "smoke_cells": smoke_cells, "source_split": strategy,
        "source_cells": len(final_train), "test_cells": len(test_run),
        "inner_cells": {k: len(v) for k,v in sampled.items()},
        "final_isoforms": len(final_yi), "final_genes": len(np.unique(dataset.isoform_genes[final_yi])),
        "inner_isoforms": len(inner_yi), "inner_genes": len(np.unique(dataset.isoform_genes[inner_yi])),
        "evaluation_grouping": grouping, "overlapping_donors": sorted(src_donors & test_donors),
        "source_ids": dataset.obs.index[final_train].tolist(), "test_ids": dataset.obs.index[test_run].tolist(),
        "inner_ids": {k: dataset.obs.index[v].tolist() for k,v in sampled.items()},
        "data_metadata": dataset.metadata,
        "input_hashes": {f: file_hash(Path(path)/f) for f in ["obs.csv", "genes.csv", "isoforms.csv", "x_gene_counts.npz", "y_isoform_counts.npz", "dataset.json"]}}
    assert not set(manifest["source_ids"]) & set(manifest["test_ids"])
    assert set(manifest["source_ids"]) == set(sum(manifest["inner_ids"].values(), []))
    manifest["prepared_hashes"] = {str(p.relative_to(folder)): file_hash(p) for p in sorted(folder.rglob("*")) if p.is_file()}
    write_json(folder/"manifest.json", manifest)
    print({k:manifest[k] for k in ["task", "source_cells", "test_cells", "final_genes", "inner_genes"]}, flush=True)
    return folder


def load_view(folder):
    folder = Path(folder)
    view = {"folder": folder, **json.loads((folder/"catalogue.json").read_text(encoding="utf-8"))}
    view["groups"] = np.asarray(view["groups"], dtype=np.int64)
    # test_truth is intentionally not loaded here.
    for path in folder.glob("*.npy"):
        view[path.stem] = np.load(path, mmap_mode="r")
    return view
