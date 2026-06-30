"""Read-only poller: score unvetted SatNOGS observations into the store. Never POSTs."""

from __future__ import annotations

import logging
from typing import Callable, Iterable

from satnogs_signal.shared.images import crop_waterfall, load_image
from satnogs_signal.shared.satnogs_api import iter_observations
from satnogs_signal.service import store

_log = logging.getLogger(__name__)


def poll(
    norads: Iterable[int],
    conn,
    score_fn: Callable[[list], list],
    fetch_bytes: Callable[[str], bytes],
    *,
    fetch_obs=iter_observations,
    token: str | None = None,
    max_pages: int = 4,
    request_interval: float = 1.0,
    now: str = "",
) -> int:
    scored = 0
    for norad in norads:
        candidates = [
            o
            for o in fetch_obs(
                norad_cat_id=norad,
                token=token,
                request_interval=request_interval,
                max_pages=max_pages,
            )
            if o.waterfall_status == "unknown"
            and o.waterfall
            and o.ground_station is not None
        ]
        if not candidates:
            continue
        seen = store.already_scored(conn, [o.id for o in candidates])
        fresh = [o for o in candidates if o.id not in seen]

        images, kept = [], []
        for o in fresh:
            try:
                images.append(crop_waterfall(load_image(fetch_bytes(o.waterfall))))
                kept.append(o)
            except Exception as e:
                _log.warning(
                    "skipping obs %s: could not fetch/decode waterfall: %s", o.id, e
                )
                continue  # per-observation isolation: skip unfetchable/corrupt
        if not images:
            continue

        try:
            probs = list(score_fn(images))
        except Exception as e:
            # batch scoring failed -> score one at a time so a single bad image
            # can't drop the whole satellite's batch; failures become None and are skipped.
            _log.warning(
                "batch scoring failed for norad %s (%s); falling back to per-image",
                norad,
                e,
            )
            probs = []
            for o, img in zip(kept, images):
                try:
                    probs.append(score_fn([img])[0])
                except Exception as e2:
                    _log.warning("skipping obs %s: scoring failed: %s", o.id, e2)
                    probs.append(None)

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
                    "scored_at": now,
                },
            )
            scored += 1
    return scored
