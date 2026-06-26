"""Assemble the gold waterfall dataset and push it to the HF Hub."""
from __future__ import annotations

from typing import Callable

from datasets import ClassLabel, Dataset, DatasetDict, Features, Image as HFImage, Value

from satnogs_signal.shared.images import load_image, preprocess
from satnogs_signal.shared.labels import is_gold, label_to_int
from satnogs_signal.data.splits import SplitConfig, dedup_by_image_hash, partition

REPO_ID = "ryroeu/satnogs-signal-waterfalls"
_LABEL_NAMES = ["without-signal", "with-signal"]  # index == integer label


def build_records(observations, fetch_bytes: Callable[[str], bytes]) -> list:
    records = []
    for o in observations:
        if not is_gold(o.waterfall_status) or not o.waterfall:
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
                "image_bytes": raw,
            }
        )
    return records


def _features() -> Features:
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
    return {r["id"]: r for r in records}


def to_dataset_dict(records: list, cfg: SplitConfig) -> DatasetDict:
    # Global exact-image dedup BEFORE partition: guarantees a byte-identical image
    # can never land in two splits (the strongest leakage guard). Per-split dedup
    # would NOT catch a cross-split duplicate, so it must happen here, before splitting.
    records = dedup_by_image_hash(records)
    by_id = _obs_index(records)

    # Rebuild Observation-like objects only need the split keys; reuse the record dicts.
    class _O:
        def __init__(self, r):
            self.norad_cat_id = r["norad"]
            self.ground_station = r["station"]
            self.start = r["timestamp"]
            self.id = r["id"]

    parts = partition([_O(r) for r in records], cfg)
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
    dataset_dict.push_to_hub(repo_id, private=True)
