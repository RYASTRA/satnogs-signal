"""Evaluate the model vs the classical baseline on a split, ranking-first and sliced."""

from __future__ import annotations

from satnogs_signal.model.baseline import score_images
from satnogs_signal.model.metrics import precision_at_k, pr_auc, roc_auc, sliced_report


def _summary(labels, scores) -> dict:
    k = min(10, len(scores))
    return {
        "roc_auc": roc_auc(labels, scores),
        "pr_auc": pr_auc(labels, scores),
        "precision_at_10": precision_at_k(labels, scores, k),
    }


def evaluate_split(split, model_scores) -> dict:
    labels = list(split["label"])
    baseline_scores = score_images(split["image"])
    sliced = {}
    for key in ("mode", "station", "norad"):
        if key in split.column_names:
            sliced[key] = sliced_report(labels, model_scores, list(split[key]))
    return {
        "model": _summary(labels, model_scores),
        "baseline": _summary(labels, baseline_scores),
        "sliced_by": sliced,
    }
