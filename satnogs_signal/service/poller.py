"""Read-only poller: score unvetted SatNOGS observations into the store. Never POSTs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable

from satnogs_signal.shared.images import crop_waterfall, load_image
from satnogs_signal.shared.satnogs_api import (
    FetchPolicy,
    Observation,
    iter_observations,
)
from satnogs_signal.service import store

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PollConfig:
    """Fetch + scoring knobs for a poll run (see :func:`poll`)."""

    token: str | None = None
    max_pages: int = 4
    request_interval: float = 1.0
    now: str = ""
    fetch_obs: Callable[..., Iterable[Observation]] = iter_observations


def _candidates(norad: int, config: PollConfig) -> list:
    """Fetch a satellite's unvetted, image- and station-bearing observations."""
    return [
        o
        for o in config.fetch_obs(
            norad_cat_id=norad,
            max_pages=config.max_pages,
            policy=FetchPolicy(
                token=config.token, request_interval=config.request_interval
            ),
        )
        if o.waterfall_status == "unknown"
        and o.waterfall
        and o.ground_station is not None
    ]


def _download_images(fresh, fetch_bytes: Callable[[str], bytes]):
    """Fetch + decode each observation's waterfall, skipping (logging) bad ones.

    Per-observation isolation: an unfetchable or corrupt waterfall is logged and
    dropped so it can't sink the rest of the satellite's batch. ``requests`` network
    errors and PIL decode errors are both ``OSError`` subclasses.
    """
    images, kept = [], []
    for o in fresh:
        try:
            images.append(crop_waterfall(load_image(fetch_bytes(o.waterfall))))
            kept.append(o)
        except (OSError, ValueError) as e:
            _log.warning(
                "skipping obs %s: could not fetch/decode waterfall: %s", o.id, e
            )
    return images, kept


def _score(
    score_fn: Callable[[list], list], images: list, kept: list, norad: int
) -> list:
    """Score images as a batch; on failure fall back to per-image so one bad image
    can't drop the whole satellite's batch (failures become None and are skipped)."""
    try:
        return list(score_fn(images))
    except (RuntimeError, ValueError) as e:
        _log.warning(
            "batch scoring failed for norad %s (%s); falling back to per-image",
            norad,
            e,
        )
        probs = []
        for o, img in zip(kept, images):
            try:
                probs.append(score_fn([img])[0])
            except (RuntimeError, ValueError) as e2:
                _log.warning("skipping obs %s: scoring failed: %s", o.id, e2)
                probs.append(None)
        return probs


def _store_scores(conn, kept: list, probs: list, config: PollConfig) -> int:
    """Persist non-None scores for kept observations; return how many were stored."""
    stored = 0
    for o, p in zip(kept, probs):
        if p is None:
            continue  # this observation's scoring failed; skip it, keep the rest
        store.upsert_prediction(
            conn,
            {
                "obs_id": o.id,
                "norad": o.norad_cat_id,
                "mode": o.transmitter_mode,
                "station": o.ground_station,
                "timestamp": o.start,
                "waterfall_url": o.waterfall,
                "p_signal": float(p),
                "predicted_label": int(float(p) >= 0.5),
                "scored_at": config.now,
            },
        )
        stored += 1
    return stored


def poll(
    norads: Iterable[int],
    conn,
    score_fn: Callable[[list], list],
    fetch_bytes: Callable[[str], bytes],
    config: PollConfig = PollConfig(),
) -> int:
    """Score unvetted observations for each NORAD into the store; return count scored."""
    scored = 0
    for norad in norads:
        candidates = _candidates(norad, config)
        if not candidates:
            continue
        seen = store.already_scored(conn, [o.id for o in candidates])
        fresh = [o for o in candidates if o.id not in seen]
        images, kept = _download_images(fresh, fetch_bytes)
        if not images:
            continue
        probs = _score(score_fn, images, kept, norad)
        scored += _store_scores(conn, kept, probs, config)
    return scored
