"""Audit the live network to pick narrow-and-deep candidate satellites."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from satnogs_signal.shared.labels import is_gold
from satnogs_signal.shared.satnogs_api import Observation, iter_observations


def tally_candidates(observations: Iterable[Observation]) -> dict:
    """Count gold with-signal/without-signal observations and modes per satellite."""
    tally: dict = defaultdict(
        lambda: {"with-signal": 0, "without-signal": 0, "modes": set()}
    )
    for o in observations:
        status = o.waterfall_status
        if status is None or not is_gold(status):
            continue
        counts = tally[o.norad_cat_id]
        counts[status] += 1
        if o.transmitter_mode:
            counts["modes"].add(o.transmitter_mode)
    return dict(tally)


def rank_candidates(tally: dict, min_per_class: int) -> list:
    """Keep satellites with enough of both classes, ranked by total observations."""
    qualifying = [
        (norad, e)
        for norad, e in tally.items()
        if e["with-signal"] >= min_per_class and e["without-signal"] >= min_per_class
    ]
    qualifying.sort(
        key=lambda kv: kv[1]["with-signal"] + kv[1]["without-signal"], reverse=True
    )
    return qualifying


def run_audit(pages_per_class: int = 8, min_per_class: int = 150, session=None) -> list:
    """Sample both classes from the live network and return ranked candidates."""
    sampled = []
    for status in ("with-signal", "without-signal"):
        sampled.extend(
            iter_observations(
                waterfall_status=status, session=session, max_pages=pages_per_class
            )
        )
    return rank_candidates(tally_candidates(sampled), min_per_class)


if __name__ == "__main__":  # pragma: no cover
    for norad, entry in run_audit():
        print(
            norad, entry["with-signal"], entry["without-signal"], sorted(entry["modes"])
        )
