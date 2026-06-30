"""Load the gold waterfall DatasetDict and derive class weights for imbalance."""

from __future__ import annotations

import os
from collections import Counter

from datasets import DatasetDict, load_dataset, load_from_disk


def load_splits(source: str) -> DatasetDict:
    if os.path.isdir(source):
        return load_from_disk(source)
    return load_dataset(source)


def class_weights(dataset) -> list:
    counts = Counter(dataset["label"])
    n = sum(counts.values())
    # inverse-frequency, normalized so the common class ~1.0
    return [n / (2 * counts.get(c, 1)) for c in (0, 1)]
