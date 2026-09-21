"""Resumable unified matrix-only benchmark. External truth enters evaluate only.

Run: python -m short2long.unified_benchmark --output-root ... [--smoke]
The small smoke uses full catalogues but never becomes a main-table result.
"""
from __future__ import annotations
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from .conservative_model import (ConservativeIsoBudget, cell_targets, macro_cross_entropy,
                                 macro_jsd, choose_alpha)
from .conservative_train import ALPHAS, cluster_targets
from .matrix_benchmark import _training_prior, knn_probabilities
from .published_benchmark import (amp_context, batches, make_optimizer, check_loss,
                                  file_hash, write_json, verify_upstream)
from .published_models import AllocationAdapter, build_published_model
from .train import set_seed
from .unified_data import SPECS, SPLIT_SEED, prepare, load_view
from .unified_metrics import evaluate

CLASSICAL = ["Mean", "PCA-kNN", "PCA-Ridge"]
NEURAL = ["MLP", "Context-MLP", "TabResNet", "FT-Transformer", "BABEL", "scButterfly", "IsoBudget-v2"]
METHODS = CLASSICAL + NEURAL
PROTOCOL = Path("reports/unified_benchmark_protocol_20260908.md")


class PlainMLP(AllocationAdapter):
    def __init__(self, n_input, groups):
        super().__init__(groups)
        self.network = nn.Sequential(nn.Linear(n_input, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(.15), nn.Linear(256, 256), nn.GELU(), nn.Dropout(.15), nn.Linear(256, len(groups)))

    def forward(self, x):
        return self.proportions(self.network(x))


def tensor(a, device):
    return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)


def build_model(method, view, name, device, prior=None, reference=None):
    n = view[f"{name}_x"].shape[1]
    if method == "IsoBudget-v2":
        if prior is None:
            prior = _training_prior(sparse.csr_matrix(view[f"{name}_y"]), view["groups"])
        if reference is None:
            reference = view[f"{name}_x"].mean(0)
        return ConservativeIsoBudget(n, view["groups"], view["structures"], prior, reference).to(device)
    if method in ["MLP", "Context-MLP"]:
        return PlainMLP(n*(2 if method == "Context-MLP" else 1), view["groups"]).to(device)
    return build_published_model(method, n, view["groups"], view["structures"]).to(device)


def inputs(view, name, ix, method, device):
    x = tensor(view[f"{name}_x"][ix], device)
    if method == "Context-MLP":
        x = torch.cat((x, tensor(view[f"{name}_context"][ix], device)), 1)
    return x


def forward(model, method, view, name, ix, device):
    x = inputs(view, name, ix, method, device)
    if method == "IsoBudget-v2":
        return model(x, tensor(view[f"{name}_context"][ix], device))
    with amp_context(method, device):
        return model(x).float()


@torch.no_grad()
def validation(model, method, view, batch_size, device):
    model.eval()
    groups = torch.as_tensor(view["groups"], device=device)
    total, count = 0., 0
    for ix in batches(len(view["val_x"]), batch_size):
        p = forward(model, method, view, "val", ix, device)
        value, n = macro_jsd(p, tensor(view["val_y"][ix], device), groups)
        total += float(value); count += int(n)
    if not count:
        raise ValueError("No evaluable source validation cell-gene")
    return total/count


def fit(method, view, args, seed, lr, folder, tag, epochs=None):
    device = torch.device("cuda:0")
    size = 8 if method == "FT-Transformer" else 64
    name = "train" if epochs is not None else "fit"
    state_path, record_path = folder/f"{tag}.pt", folder/f"{tag}.json"
    if record_path.exists() and state_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        model = build_model(method, view, name, device, state.get("prior"), state.get("reference_expression"))
        model.load_state_dict(state)
        return model, record
    set_seed(seed)
    model = build_model(method, view, name, device)
    optimizer = (torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) if method == "IsoBudget-v2"
                 else make_optimizer(model, method, lr))
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    history, pre_history = [], []
    if method == "scButterfly":
        for side in ["x", "y"]:
            for epoch in range(1, args.pretrain_epochs+1):
                model.train(); losses = []
                for ix in batches(len(view[f"{name}_x"]), size, rng):
                    optimizer.zero_grad(set_to_none=True)
                    loss = model.pretrain_loss(inputs(view, name, ix, method, device),
                        tensor(view[f"{name}_y"][ix], device), side, .01*min(1, epoch/10))
                    check_loss(loss, f"{tag} pretrain {side}")
                    loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
                    losses.append(float(loss.detach()))
                event = {"side": side, "epoch": epoch, "loss": float(np.mean(losses)), "seconds": time.perf_counter()-started}
                pre_history.append(event)
                write_json(folder/f"progress_{tag}_pretrain.json", pre_history)
                print({"method": method, "stage": tag, **event}, flush=True)
        optimizer = make_optimizer(model, method, lr)
        disc_optimizer = torch.optim.SGD(model.discriminators.parameters(), lr=5*lr)
    if method == "IsoBudget-v2":
        cy, cv = cluster_targets(view[f"{name}_y"], view[f"{name}_context_labels"], view["groups"], model.prior.detach().cpu().numpy())
        cx = tensor(view[f"{name}_context_means"], device)
        ct = tensor(cy, device); cm = torch.as_tensor(cv, device=device)
        del cy, cv
    best, selected_epoch, stale, best_state = float("inf"), 0, 0, None
    for epoch in range(1, (epochs if epochs is not None else args.epochs)+1):
        model.train(); loss_sum, num = 0., 0
        for ix in batches(len(view[f"{name}_x"]), size, rng):
            xb = inputs(view, name, ix, method, device)
            yb = tensor(view[f"{name}_y"][ix], device)
            if method == "scButterfly":
                disc_optimizer.zero_grad(set_to_none=True)
                loss_d = model.discriminator_loss(xb, yb)
                check_loss(loss_d, "discriminator")
                loss_d.backward(); nn.utils.clip_grad_norm_(model.discriminators.parameters(), 5); disc_optimizer.step()
                for parameter in model.discriminators.parameters(): parameter.requires_grad_(False)
            optimizer.zero_grad(set_to_none=True)
            if method == "IsoBudget-v2":
                p = model(xb, tensor(view[f"{name}_context"][ix], device), calibrated=False)
                target, valid = cell_targets(yb, model.isoform_gene_index, model.prior)
                cell_loss = macro_cross_entropy(p, target, valid, model.isoform_gene_index)
                cix = torch.as_tensor(rng.choice(len(cx), min(8,len(cx)), replace=False), device=device)
                cp = model(cx[cix], cx[cix], calibrated=False)
                cluster_loss = macro_cross_entropy(cp, ct[cix], cm[cix], model.isoform_gene_index)
                anchor = (p*(p.clamp_min(1e-8).log()-model.prior.log())).sum()/(len(p)*model.n_target_genes)
                loss = cell_loss + .5*cluster_loss + .02*anchor
            else:
                with amp_context(method, device):
                    loss, _ = model.loss(xb, yb, tensor(view[f"{name}_raw"][ix], device), kl_weight=.01*min(1,epoch/10))
            check_loss(loss, tag)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1 if method == "IsoBudget-v2" else 5)
            optimizer.step()
            if method == "scButterfly":
                for parameter in model.discriminators.parameters(): parameter.requires_grad_(True)
            loss_sum += float(loss.detach()); num += 1
        score = validation(model, method, view, size, device) if epochs is None else None
        if score is not None:
            if score < best-1e-7:
                best, selected_epoch, stale = score, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                stale += 1
        event = {"epoch": epoch, "loss": loss_sum/num, "validation_jsd": score, "seconds": time.perf_counter()-started}
        history.append(event); write_json(folder/f"progress_{tag}.json", history)
        print({"method": method, "seed": seed, "stage": tag, **event}, flush=True)
        if score is not None and stale >= args.patience:
            break
    if epochs is None:
        if best_state is None: raise ValueError("No finite validation checkpoint")
        model.load_state_dict(best_state)
    else:
        selected_epoch = epoch
    record = {"learning_rate": lr, "selected_epoch": selected_epoch, "validation_jsd": best if epochs is None else None,
              "history": history, "pretrain": pre_history, "seconds": time.perf_counter()-started,
              "microbatch": size, "parameters": sum(p.numel() for p in model.parameters()),
              "peak_gpu_mb": torch.cuda.max_memory_allocated(device)/1024**2,
              "selected_at_budget_boundary": epochs is None and selected_epoch == args.epochs}
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, state_path)
    write_json(record_path, record)
    return model, record


@torch.no_grad()
def calibrate(model, view):
    model.eval(); device = torch.device("cuda:0")
    groups = torch.as_tensor(view["groups"], device=device)
    sums, n = np.zeros(len(ALPHAS)), 0
    for ix in batches(len(view["cal_x"]),64):
        p = forward(model, "IsoBudget-v2", view, "cal", ix, device)
        y = tensor(view["cal_y"][ix], device)
        for j,a in enumerate(ALPHAS):
            val, count = macro_jsd((1-a)*model.prior+a*p, y, groups)
            sums[j] += float(val)
        n += int(count)
    if not n: raise ValueError("No source calibration cell-gene")
    records = [{"alpha":a, "macro_jsd":s/n} for a,s in zip(ALPHAS,sums)]
    return choose_alpha(records), records


def fit_ridge(embedding, y, groups, alpha):
    y = sparse.csc_matrix(y)
    prior = _training_prior(y, groups)
    coef = np.zeros((y.shape[1], embedding.shape[1]), np.float32)
    intercept = np.log(prior).astype(np.float32)
    fitted = 0
    for gene in np.unique(groups):
        ix = np.flatnonzero(groups == gene)
        counts = y[:,ix].toarray(); totals = counts.sum(1); valid = totals > 0
        if valid.sum() < 10: continue
        target = np.log((counts[valid]+.5)/(totals[valid,None]+.5*len(ix)))
        target -= target.mean(1,keepdims=True)
        model = Ridge(alpha=alpha, solver="lsqr").fit(embedding[valid], target, sample_weight=np.minimum(totals[valid],10))
        coef[ix], intercept[ix] = model.coef_, model.intercept_; fitted += 1
    return {"coef":coef, "intercept":intercept, "fitted_genes":fitted}


def classical_predict(model, x):
    groups = model["groups"]
    if model["method"] == "Mean":
        return np.broadcast_to(model["prior"], (len(x), len(groups)))
    embedding = model["pca"].transform(x).astype(np.float32)
    if model["method"] == "PCA-kNN":
        return knn_probabilities(model["embedding"], embedding, model["y"], groups,
                                 model["k"], model["prior_strength"])[0]
    scores = embedding @ model["coef"].T + model["intercept"]
    for gene in np.unique(groups):
        ix = np.flatnonzero(groups == gene)
        block = scores[:,ix]; block -= block.max(1,keepdims=True)
        block = np.exp(np.clip(block,-30,0)); scores[:,ix] = block/block.sum(1,keepdims=True)
    return scores


def classical_fit(method, view, name, params):
    x, y, groups = view[f"{name}_x"], sparse.csr_matrix(view[f"{name}_y"]), view["groups"]
    model = {"method":method, "groups":groups, "prior":_training_prior(y,groups), **params}
    if method != "Mean":
        pca = PCA(n_components=min(32,len(x)-1,x.shape[1]),svd_solver="randomized",random_state=SPLIT_SEED)
        embedding = pca.fit_transform(x).astype(np.float32)
        model["pca"] = pca
        if method == "PCA-kNN": model.update(embedding=embedding, y=y)
        else: model.update(fit_ridge(embedding,y,groups,params["ridge_alpha"]))
    return model


def classical_score(model, view):
    total, count = 0., 0
    device = torch.device("cuda:0")
    groups = torch.as_tensor(view["groups"],device=device)
    for ix in batches(len(view["val_x"]),256):
        value, n = macro_jsd(tensor(classical_predict(model,view["val_x"][ix]),device),tensor(view["val_y"][ix],device),groups)
        total += float(value); count += int(n)
    if not count: raise ValueError("No validation coverage")
    return total/count


@torch.no_grad()
def save_predictions(model, method, view, folder):
    device = torch.device("cuda:0")
    if method in NEURAL: model.eval()
    shape = (len(view["test_x"]),len(view["groups"]))
    p = np.lib.format.open_memmap(folder/"probabilities.npy",mode="w+",dtype=np.float32,shape=shape)
    size = 8 if method == "FT-Transformer" else 64 if method in NEURAL else 256
    for ix in batches(shape[0],size):
        p[ix] = (forward(model,method,view,"test",ix,device).cpu().numpy() if method in NEURAL
                 else classical_predict(model,view["test_x"][ix]))
    p.flush()
    return p


def run_one(task, method, seed, args):
    folder = args.output_root/task/method/(f"seed_{seed}" if method in NEURAL else "deterministic")
    if (folder/"result.json").exists():
        print(f"Already complete: {folder}",flush=True); return
    folder.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    prepared = getattr(args, "prepared_root", args.output_root)/"prepared"/task
    inner, final = load_view(prepared/"inner"), load_view(prepared/"final")
    selection_path = folder/"selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
    elif method in NEURAL:
        records = []
        for lr in args.learning_rates:
            model, record = fit(method,inner,args,seed,lr,folder,f"tune_{lr:g}")
            records.append(record); del model; gc.collect(); torch.cuda.empty_cache()
        best = min(records,key=lambda r:r["validation_jsd"])
        selection = {"tuning":records,"best":best,"alpha":1.,"calibration":[]}
        if method == "IsoBudget-v2":
            model, _ = fit(method,inner,args,seed,best["learning_rate"],folder,f"tune_{best['learning_rate']:g}")
            selection["alpha"], selection["calibration"] = calibrate(model,inner)
            del model; gc.collect(); torch.cuda.empty_cache()
        write_json(selection_path,selection)
    else:
        candidates = ([{}] if method == "Mean" else
            [{"k":k,"prior_strength":s} for k in [15,50] for s in [1.,5.]] if method == "PCA-kNN" else
            [{"ridge_alpha":a} for a in [1.,10.,100.,1000.]])
        records = []
        for params in candidates:
            model = classical_fit(method,inner,"fit",params)
            score = classical_score(model,inner); records.append({"parameters":params,"validation_jsd":score})
            print({"task":task,"method":method,**records[-1]},flush=True)
            del model; gc.collect()
        selection = {"tuning":records,"best":min(records,key=lambda r:r["validation_jsd"])}
        write_json(selection_path,selection)
    del inner; gc.collect()
    if method in NEURAL:
        best = selection["best"]
        model, refit = fit(method,final,args,seed,best["learning_rate"],folder,"refit",epochs=best["selected_epoch"])
        if method == "IsoBudget-v2": model.calibration_alpha.fill_(selection["alpha"])
        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},folder/"model.pt")
    else:
        model = classical_fit(method,final,"train",selection["best"]["parameters"])
        refit = {"fitted_genes":model.get("fitted_genes"),"parameters":selection["best"]["parameters"]}
        with (folder/"model.pkl").open("wb") as stream: pickle.dump(model,stream)
    probabilities = save_predictions(model,method,final,folder)
    del model; gc.collect(); torch.cuda.empty_cache()
    # Audit disk checkpoint: inference is repeated with SR inputs only.
    if method in NEURAL:
        state = torch.load(folder/"model.pt",map_location="cpu",weights_only=True)
        restored = build_model(method,final,"train",torch.device("cuda:0"),state.get("prior"),state.get("reference_expression"))
        restored.load_state_dict(state); restored.eval()
    else:
        with (folder/"model.pkl").open("rb") as stream: restored = pickle.load(stream)
    error = 0.
    with torch.no_grad():
        for ix in batches(min(32,len(probabilities)),8):
            repeated = (forward(restored,method,final,"test",ix,torch.device("cuda:0")).cpu().numpy()
                        if method in NEURAL else classical_predict(restored,final["test_x"][ix]))
            error = max(error,float(np.max(np.abs(repeated-probabilities[ix]))))
    if error > 1e-4:
        raise ValueError(f"Checkpoint round-trip mismatch: {error}")
    del restored; gc.collect(); torch.cuda.empty_cache()
    metrics = evaluate(final,probabilities)
    np.savez_compressed(folder/"prediction_index.npz",cell_ids=final["test_ids"],isoforms=np.asarray(final["isoforms"]),
                        cluster_labels=final["evaluation_clusters"])
    result = {"task":task,"method":method,"seed":seed if method in NEURAL else None,"smoke":args.smoke,
              "selection":selection,"refit":refit,"metrics":metrics,"seconds":time.perf_counter()-started,
              "checkpoint_roundtrip_max_abs":error,"manifest_sha256":file_hash(prepared/"manifest.json"),
              "prediction_sha256":file_hash(folder/"probabilities.npy"),
              "checkpoint_sha256":file_hash(folder/("model.pt" if method in NEURAL else "model.pkl"))}
    write_json(folder/"result.json",result)
    print({"completed":str(folder),"metrics":{k:v for k,v in metrics.items() if not isinstance(v,list)}},flush=True)


def summarize(root):
    rows = []
    for path in sorted(root.glob("*/*/*/result.json")):
        row = json.loads(path.read_text(encoding="utf-8")); row["path"] = str(path); rows.append(row)
    write_json(root/"summary.json",{"completed_runs":len(rows),"runs":rows})
    fields = ["confident_dominant_isoform_gene_macro_accuracy","cell_gene_1_minus_jsd",
              "cluster_allocation_spearman_median","spatial_proportion_moran_spearman","cluster_contrast_mae"]
    lines = ["# Unified benchmark run results", "", "Only completed and validated runs are included. NA is not zero. Random seeds are not independent biological replicates. Definitions follow the prespecified protocol.", "",
             "| Setting | Method | Seed / deterministic | Dominant accuracy ↑ | Cell-gene 1−JSD ↑ | Cluster Spearman ↑ | Spatial Moran ↑ | Cluster-contrast MAE ↓ |",
             "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        vals = ["NA" if row["metrics"].get(k) is None else f"{row['metrics'][k]:.6f}" for k in fields]
        lines.append("| "+" | ".join([row["task"],row["method"],str(row["seed"]) if row["seed"] is not None else "Deterministic",*vals])+" |")
    (root/"results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--tasks",nargs="+",choices=list(SPECS),default=list(SPECS))
    parser.add_argument("--methods",nargs="+",choices=METHODS,default=METHODS)
    parser.add_argument("--seeds",nargs="+",type=int,default=[2026,2027,2028])
    parser.add_argument("--epochs",type=int,default=50)
    parser.add_argument("--patience",type=int,default=8)
    parser.add_argument("--pretrain-epochs",type=int,default=10)
    parser.add_argument("--learning-rates",nargs="+",type=float,default=[.0003,.001])
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--prepare-only",action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; CPU fallback forbidden")
    torch.set_num_threads(4)
    threadpool_limits(4)
    if args.smoke:
        args.epochs, args.patience, args.pretrain_epochs, args.seeds = 2, 2, 1, [2026]
    args.output_root.mkdir(parents=True,exist_ok=True)
    config = {"arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k != "prepare_only"},
              "protocol_sha256":file_hash(PROTOCOL),"torch":torch.__version__,"python":sys.version,
              "gpu":torch.cuda.get_device_name(0),"device":"cuda:0","CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),
              "code_hashes":{p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},"upstream":verify_upstream()}
    config_path = args.output_root/"run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != config: raise ValueError("Config/code changed; use a new output root, never silently mix runs")
    else: write_json(config_path,config)
    for task in args.tasks: prepare(task,args.output_root,128 if args.smoke else None)
    if args.prepare_only: return
    queue = [(task,method,seed) for seed in args.seeds for task in args.tasks for method in args.methods
             if method in NEURAL or seed == args.seeds[0]]
    failures = []
    for ordinal,(task,method,seed) in enumerate(queue,1):
        write_json(args.output_root/"status.json",{"state":"running","pid":os.getpid(),"position":ordinal,"total":len(queue),
            "current":{"task":task,"method":method,"seed":seed},"failures":failures,"updated":time.time()})
        try: run_one(task,method,seed,args)
        except Exception:
            failure = {"task":task,"method":method,"seed":seed,"error":traceback.format_exc()}
            failures.append(failure)
            write_json(args.output_root/"failures"/f"{task}_{method}_{seed}.json",failure)
            print(failure,flush=True)
        finally:
            gc.collect(); torch.cuda.empty_cache(); summarize(args.output_root)
    completed = summarize(args.output_root)
    write_json(args.output_root/"status.json",{"state":"complete" if not failures else "complete_with_failures",
        "completed":completed,"total":len(queue),"failures":failures,"updated":time.time()})
    if failures: raise SystemExit(1)


if __name__ == "__main__": main()
