"""Gold waterfall-vetting label mapping. Never use observation `status` as a label."""

from __future__ import annotations

from typing import Optional

GOLD_LABELS = {"without-signal": 0, "with-signal": 1}


def is_gold(waterfall_status: Optional[str]) -> bool:
    """True if the observation carries a gold with/without-signal vetting label."""
    return waterfall_status in GOLD_LABELS


def label_to_int(waterfall_status: str) -> int:
    """Map a gold ``waterfall_status`` string to its integer class label."""
    return GOLD_LABELS[waterfall_status]
