"""Ranking-first evaluation metrics. Accuracy is intentionally NOT here — the
service ranks observations, so we score separation quality, not labeling."""

from __future__ import annotations

from collections import defaultdict

from sklearn.metrics import average_precision_score, roc_auc_score


def roc_auc(labels, scores) -> float:
    """Area under the ROC curve for binary labels given real-valued scores."""
    return float(roc_auc_score(labels, scores))


def pr_auc(labels, scores) -> float:
    """Area under the precision-recall curve (average precision)."""
    return float(average_precision_score(labels, scores))


def precision_at_k(labels, scores, k: int) -> float:
    """Fraction of positives among the ``k`` highest-scored items."""
    if k <= 0:
        raise ValueError("k must be > 0")
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    top = [labels[i] for i in order]
    return sum(top) / len(top)


def sliced_report(labels, scores, groups) -> dict:
    """Per-group sample count and ROC-AUC (None when a group is single-class)."""
    by_group = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    out = {}
    for g, ids in by_group.items():
        gl = [labels[i] for i in ids]
        gs = [scores[i] for i in ids]
        out[g] = {
            "n": len(ids),
            "roc_auc": roc_auc(gl, gs) if len(set(gl)) > 1 else None,
        }
    return out
