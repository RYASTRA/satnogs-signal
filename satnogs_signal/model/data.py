"""Load the gold waterfall DatasetDict and derive class weights for imbalance."""

from __future__ import annotations

import os
from collections import Counter

from datasets import DatasetDict, load_dataset, load_from_disk


def load_splits(source: str) -> DatasetDict:
    """Load splits from a local disk path or the Hub as a DatasetDict."""
    if os.path.isdir(source):
        loaded = load_from_disk(source)
    else:
        loaded = load_dataset(source)
    assert isinstance(loaded, DatasetDict), "expected a DatasetDict with named splits"
    return loaded


def class_weights(dataset) -> list:
    """Return inverse-frequency weights for the binary label, normalized."""
    counts = Counter(dataset["label"])
    n = sum(counts.values())
    # inverse-frequency, normalized so the common class ~1.0
    return [n / (2 * counts.get(c, 1)) for c in (0, 1)]
