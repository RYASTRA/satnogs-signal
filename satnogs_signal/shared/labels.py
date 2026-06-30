"""Gold waterfall-vetting label mapping. Never use observation `status` as a label."""

from __future__ import annotations

from typing import Optional

GOLD_LABELS = {"without-signal": 0, "with-signal": 1}


def is_gold(waterfall_status: Optional[str]) -> bool:
    return waterfall_status in GOLD_LABELS


def label_to_int(waterfall_status: str) -> int:
    return GOLD_LABELS[waterfall_status]
