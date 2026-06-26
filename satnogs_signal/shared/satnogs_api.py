"""Read-only client for the SatNOGS Network API."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    import requests

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


def _retry_after_seconds(resp, default: float, cap: float) -> float:
    """Honor the server's Retry-After header (in seconds) if present, capped; else default."""
    ra = resp.headers.get("Retry-After") if hasattr(resp, "headers") else None
    if ra is not None:
        try:
            return min(float(ra), cap)
        except (TypeError, ValueError):
            return default
    return default


def _get_with_backoff(
    session,
    url,
    params,
    max_retries,
    backoff_base,
    sleep,
    headers=None,
    max_retry_after: float = 1800.0,
) -> "requests.Response":
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    for attempt in range(max_retries):
        resp = session.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp
        if resp.status_code in _RETRY_STATUS and attempt < max_retries - 1:
            wait = _retry_after_seconds(
                resp, default=backoff_base * (2 ** attempt), cap=max_retry_after
            )
            sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError("unreachable: loop always returns or raises")


def iter_observations(
    *,
    norad_cat_id: Optional[int] = None,
    waterfall_status: Optional[str] = None,
    session=None,
    max_pages: Optional[int] = None,
    max_retries: int = 5,
    backoff_base: float = 1.0,
    sleep=time.sleep,
    token: Optional[str] = None,
    request_interval: float = 0.0,
) -> Iterator[Observation]:
    """Yield gold/any observations, following cursor pagination politely.

    Pass ``token`` to authenticate (sends ``Authorization: Token <token>``); the
    SatNOGS Network API grants authenticated callers a higher rate limit. Pass
    ``request_interval`` to pause that many seconds before each page fetch
    (proactive throttling), so we stay under the limit instead of only backing
    off after a 429.
    """
    if session is None:
        import requests

        session = requests.Session()

    headers = {"Authorization": f"Token {token}"} if token else None

    params: dict[str, object] | None = {"format": "json"}
    if norad_cat_id is not None:
        params["norad_cat_id"] = norad_cat_id
    if waterfall_status is not None:
        params["waterfall_status"] = WATERFALL_STATUS_FILTER[waterfall_status]

    url = f"{BASE_URL}/observations/"
    pages = 0
    while url:
        if request_interval:
            sleep(request_interval)
        resp = _get_with_backoff(
            session, url, params, max_retries, backoff_base, sleep, headers
        )
        params = None  # 'next' Link is an absolute URL carrying its own cursor
        for item in resp.json():
            yield parse_observation(item)
        pages += 1
        if max_pages is not None and pages >= max_pages:
            return
        url = resp.links.get("next", {}).get("url")
