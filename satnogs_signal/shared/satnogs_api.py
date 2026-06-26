"""Read-only client for the SatNOGS Network API."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional

BASE_URL = "https://network.satnogs.org/api"


@dataclass(frozen=True)
class Observation:
    id: int
    norad_cat_id: Optional[int]
    transmitter_mode: Optional[str]
    ground_station: Optional[int]
    station_name: Optional[str]
    start: Optional[str]
    waterfall: Optional[str]
    waterfall_status: Optional[str]
    status: Optional[str]


def parse_observation(d: dict) -> Observation:
    return Observation(
        id=d["id"],
        norad_cat_id=d.get("norad_cat_id"),
        transmitter_mode=d.get("transmitter_mode"),
        ground_station=d.get("ground_station"),
        station_name=d.get("station_name"),
        start=d.get("start"),
        waterfall=d.get("waterfall"),
        waterfall_status=d.get("waterfall_status"),
        status=d.get("status"),
    )


WATERFALL_STATUS_FILTER = {"without-signal": 0, "with-signal": 1}

_RETRY_STATUS = {429, 500, 502, 503, 504}


def _get_with_backoff(session, url, params, max_retries, backoff_base, sleep):
    last = None
    for attempt in range(max_retries):
        last = session.get(url, params=params, timeout=30)
        if last.status_code == 200:
            return last
        if last.status_code in _RETRY_STATUS and attempt < max_retries - 1:
            sleep(backoff_base * (2 ** attempt))
            continue
        last.raise_for_status()
    last.raise_for_status()
    return last  # pragma: no cover


def iter_observations(
    *,
    norad_cat_id: Optional[int] = None,
    waterfall_status: Optional[str] = None,
    session=None,
    max_pages: Optional[int] = None,
    max_retries: int = 5,
    backoff_base: float = 1.0,
    sleep=time.sleep,
) -> Iterator[Observation]:
    """Yield gold/any observations, following cursor pagination politely."""
    if session is None:
        import requests

        session = requests.Session()

    params = {"format": "json"}
    if norad_cat_id is not None:
        params["norad_cat_id"] = norad_cat_id
    if waterfall_status is not None:
        params["waterfall_status"] = WATERFALL_STATUS_FILTER[waterfall_status]

    url = f"{BASE_URL}/observations/"
    pages = 0
    while url:
        resp = _get_with_backoff(session, url, params, max_retries, backoff_base, sleep)
        params = None  # 'next' Link is an absolute URL carrying its own cursor
        for item in resp.json():
            yield parse_observation(item)
        pages += 1
        if max_pages is not None and pages >= max_pages:
            return
        url = resp.links.get("next", {}).get("url")
