"""Ranking-first evaluation metrics. Accuracy is intentionally NOT here — the
service ranks observations, so we score separation quality, not labeling."""
from __future__ import annotations

from sklearn.metrics import average_precision_score, roc_auc_score


def roc_auc(labels, scores) -> float:
    return float(roc_auc_score(labels, scores))


def pr_auc(labels, scores) -> float:
    return float(average_precision_score(labels, scores))


def precision_at_k(labels, scores, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    top = [labels[i] for i in order]
    return sum(top) / len(top)
