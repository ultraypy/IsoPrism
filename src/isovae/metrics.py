from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

Array = np.ndarray


def evaluate_usage_metrics(
    pred_usage: Array,
    y_usage: Array,
    y_counts: Array,
    groups: Sequence[Array],
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Evaluate predicted within-gene isoform usage against observed usage."""
    ce_num = 0.0
    ce_den = 0.0
    top1_correct = 0
    top2_correct = 0
    total = 0
    mae_list: List[float] = []
    pearson_list: List[float] = []
    cosine_list: List[float] = []

    for idx in groups:
        pred = pred_usage[:, idx]
        truth = y_usage[:, idx]
        counts = y_counts[:, idx]
        cov = counts.sum(axis=1)
        mask = cov > 0
        if not mask.any():
            continue

        ce = -(truth[mask] * np.log(pred[mask] + eps)).sum(axis=1)
        w = np.sqrt(cov[mask] + 1.0)
        ce_num += float(np.sum(ce * w))
        ce_den += float(np.sum(w))

        obs_top = np.argmax(truth[mask], axis=1)
        pred_order = np.argsort(pred[mask], axis=1)[:, ::-1]
        top1_correct += int(np.sum(pred_order[:, 0] == obs_top))
        top2_correct += int(np.sum([obs_top[i] in pred_order[i, :2] for i in range(len(obs_top))]))
        total += int(mask.sum())
        mae_list.append(float(np.mean(np.abs(pred[mask] - truth[mask]))))

        if len(idx) >= 3:
            p = pred[mask]
            t = truth[mask]
            dot = np.sum(p * t, axis=1)
            denom = np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1) + eps
            cosine_list.extend((dot / denom).astype(float).tolist())
            for i in range(p.shape[0]):
                if np.std(p[i]) > eps and np.std(t[i]) > eps:
                    r = np.corrcoef(p[i], t[i])[0, 1]
                    if np.isfinite(r):
                        pearson_list.append(float(r))

    return {
        "weighted_ce": float(ce_num / max(ce_den, eps)),
        "top1": float(top1_correct / max(total, 1)),
        "top2": float(top2_correct / max(total, 1)),
        "group_mae": float(np.mean(mae_list)) if mae_list else float("nan"),
        "gene_cell_pearson": float(np.mean(pearson_list)) if pearson_list else float("nan"),
        "gene_cell_cosine": float(np.mean(cosine_list)) if cosine_list else float("nan"),
        "n_effective_gene_cell_pairs": int(total),
    }
