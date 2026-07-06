"""Inference-only: load the published classifier and score waterfall images.

Live product path only — no training, no datasets, no ``Trainer``. The poller imports
``load_scorer`` from here and reuses the returned closure across batches.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


def _resolve_size(processor) -> int:
    """Resolve the model's expected square input size from the image-processor config.
    Handles both {"shortest_edge": N} and {"height": H, "width": W} forms."""
    size_info = processor.size
    if "shortest_edge" in size_info:
        return size_info["shortest_edge"]
    if "height" in size_info:
        return size_info["height"]
    return 224


def load_scorer(model_dir):
    """Load the processor + model ONCE and return ``score(images) -> list[float]``
    (P(with-signal) per image). Reuse the returned callable across many batches to
    avoid reloading the model on every call (e.g. in the poller)."""
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir).eval()
    size = _resolve_size(processor)
    mean, std = processor.image_mean, processor.image_std

    def score(images) -> list:
        scores = []
        with torch.no_grad():
            for im in images:
                arr = (
                    np.asarray(im.convert("RGB").resize((size, size)), dtype=np.float32)
                    / 255.0
                )
                arr = (arr - mean) / std
                t = torch.tensor(arr).permute(2, 0, 1).float().unsqueeze(0)
                prob = torch.softmax(model(pixel_values=t).logits, dim=1)[0, 1]
                scores.append(float(prob))
        return scores

    return score
