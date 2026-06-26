"""Exact-image dedup and leakage-safe train/val/test partitioning.

Splitting by held-out STATION is the core anti-leakage measure: a station's
noise floor / RFI fingerprint must not be seen in training, or the model learns
the station instead of the signal. The held-out SATELLITE measures cross-satellite
generalization. Because a station's whole history lands in one split, same-pass
frames cannot straddle splits.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from satnogs_signal.shared.satnogs_api import Observation


def dedup_by_image_hash(records: list[dict]) -> list[dict]:
    seen: set = set()
    out: list = []
    for r in records:
        h = hashlib.sha256(r["image_bytes"]).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(r)
    return out


@dataclass
class SplitConfig:
    train_norads: list
    heldout_satellite_norad: int
    heldout_station_ids: set = field(default_factory=set)
    val_fraction_by_time: float = 0.2


def partition(observations: list[Observation], cfg: SplitConfig) -> dict[str, list[Observation]]:
    allowed = set(cfg.train_norads) | {cfg.heldout_satellite_norad}
    test, pool = [], []
    for o in observations:
        if o.norad_cat_id not in allowed:
            raise ValueError(
                f"observation {o.id} has norad {o.norad_cat_id}, which is neither a "
                f"train satellite ({cfg.train_norads}) nor the held-out satellite "
                f"({cfg.heldout_satellite_norad}); refusing to place an unexpected "
                f"satellite into a split"
            )
        if o.norad_cat_id == cfg.heldout_satellite_norad or o.ground_station in cfg.heldout_station_ids:
            test.append(o)
        else:
            pool.append(o)
    pool.sort(key=lambda o: o.start or "")
    n_val = int(len(pool) * cfg.val_fraction_by_time)
    split_at = len(pool) - n_val
    return {"train": pool[:split_at], "val": pool[split_at:], "test": test}
