"""Read-only client for the SatNOGS Network API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
