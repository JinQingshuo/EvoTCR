"""IMMREP-style unweighted macro standardized partial AUROC."""

import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate(labels, scores, targets, max_fpr=0.1):
    labels, scores, targets = map(np.asarray, (labels, scores, targets))
    if labels.ndim != 1 or labels.shape != scores.shape or labels.shape != targets.shape:
        raise ValueError("Misaligned labels / scores / target IDs")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Invalid labels or scores")
    records = []
    for t in np.unique(targets):
        ix = targets == t
        if len(np.unique(labels[ix])) != 2:
            raise ValueError(f"Target {t} lacks both classes; macro score is undefined")
        records.append(
            {
                "target": int(t),
                "full_auc": roc_auc_score(labels[ix], scores[ix]),
                "partial_auc": roc_auc_score(labels[ix], scores[ix], max_fpr=max_fpr),
            }
        )
    return {
        "macro_partial_auc": float(np.mean([r["partial_auc"] for r in records])),
        "macro_full_auc": float(np.mean([r["full_auc"] for r in records])),
        "max_fpr": max_fpr,
        "targets": records,
    }


def evaluate_five(labels, scores, targets, seeds):
    from .parameters import frozen_head

    seeds = np.asarray(seeds)
    scores = np.asarray(scores)
    if not np.array_equal(seeds, frozen_head()["final_seeds"]) or scores.shape != (5, len(labels)):
        raise ValueError("Expected the five predefined seeds and aligned prediction rows")
    records = [
        dict(seed=int(seed), **evaluate(labels, value, targets))
        for seed, value in zip(seeds, scores, strict=True)
    ]
    return {
        "per_seed": records,
        "seed_count": 5,
        "standard_deviation_ddof": 1,
        "macro_partial_auc_mean": float(np.mean([r["macro_partial_auc"] for r in records])),
        "macro_partial_auc_sd": float(np.std([r["macro_partial_auc"] for r in records], ddof=1)),
        "macro_full_auc_mean": float(np.mean([r["macro_full_auc"] for r in records])),
        "macro_full_auc_sd": float(np.std([r["macro_full_auc"] for r in records], ddof=1)),
        "ensemble_predictions_evaluated": False,
    }
