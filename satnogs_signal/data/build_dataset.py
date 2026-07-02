"""Assemble the gold waterfall dataset and push it to the HF Hub."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, cast

from datasets import ClassLabel, Dataset, DatasetDict, Features, Image as HFImage, Value

from satnogs_signal.shared.images import load_image, preprocess
from satnogs_signal.shared.labels import is_gold, label_to_int
from satnogs_signal.shared.satnogs_api import Observation
from satnogs_signal.data.splits import SplitConfig, dedup_by_image_hash, partition

REPO_ID = "ryroeu/satnogs-signal-waterfalls"
_LABEL_NAMES = ["without-signal", "with-signal"]  # index == integer label


@dataclass(frozen=True)
class _SplitKeys:
    """Minimal Observation stand-in carrying only the fields ``partition`` reads."""

    norad_cat_id: int | None
    ground_station: int | None
    start: str | None
    id: int


def build_records(observations, fetch_bytes: Callable[[str], bytes]) -> list:
    """Build dataset records from gold, image-bearing, station-bearing observations."""
    records = []
    for o in observations:
        # Skip non-gold, image-less, OR station-less observations: a null station
        # can't participate in the station-based split and breaks the int64 schema.
        if (
            not is_gold(o.waterfall_status)
            or not o.waterfall
            or o.ground_station is None
        ):
            continue
        raw = fetch_bytes(o.waterfall)
        image = preprocess(load_image(raw))
        records.append(
            {
                "id": o.id,
                "norad": o.norad_cat_id,
                "mode": o.transmitter_mode,
                "station": o.ground_station,
                "timestamp": o.start,
                "label": label_to_int(o.waterfall_status),
                "image": image,
                # Hash now and discard the raw bytes: dedup only needs the digest, so we
                # keep just the small 224x224 image + hash, not ~1-2 MB of raw PNG per obs.
                "image_hash": hashlib.sha256(raw).hexdigest(),
            }
        )
    return records


def _features() -> Features:
    """Return the HF Features schema for the dataset (image, label, and metadata columns)."""
    return Features(
        {
            "image": HFImage(),
            "label": ClassLabel(names=_LABEL_NAMES),
            "obs_id": Value("int64"),
            "norad": Value("int64"),
            "mode": Value("string"),
            "station": Value("int64"),
            "timestamp": Value("string"),
        }
    )


def _obs_index(records):
    """Map each record to its observation id for post-split lookup."""
    return {r["id"]: r for r in records}


def to_dataset_dict(records: list, cfg: SplitConfig) -> DatasetDict:
    """Dedup records globally, partition into train/val/test, and build the DatasetDict."""
    # Global exact-image dedup BEFORE partition: guarantees a byte-identical image
    # can never land in two splits (the strongest leakage guard). Per-split dedup
    # would NOT catch a cross-split duplicate, so it must happen here, before splitting.
    records = dedup_by_image_hash(records)
    by_id = _obs_index(records)

    # partition() only reads split keys, so hand it lightweight stand-ins built from
    # the record dicts (cast so the type checker accepts them where Observations go).
    parts = partition(
        cast(
            list[Observation],
            [
                _SplitKeys(
                    norad_cat_id=r["norad"],
                    ground_station=r["station"],
                    start=r["timestamp"],
                    id=r["id"],
                )
                for r in records
            ],
        ),
        cfg,
    )
    feats = _features()

    def _make(split_obs):
        rows = [by_id[o.id] for o in split_obs]
        return Dataset.from_dict(
            {
                "image": [r["image"] for r in rows],
                "label": [r["label"] for r in rows],
                "obs_id": [r["id"] for r in rows],
                "norad": [r["norad"] for r in rows],
                "mode": [r["mode"] for r in rows],
                "station": [r["station"] for r in rows],
                "timestamp": [r["timestamp"] for r in rows],
            },
            features=feats,
        )

    return DatasetDict({name: _make(parts[name]) for name in ("train", "val", "test")})


def push(dataset_dict: DatasetDict, repo_id: str = REPO_ID):  # pragma: no cover
    """Push the dataset to the HF Hub as a private repo."""
    dataset_dict.push_to_hub(repo_id, private=True)
