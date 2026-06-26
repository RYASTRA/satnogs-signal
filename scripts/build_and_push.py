"""Build the gold waterfall DatasetDict for the chosen satellites and save it locally.

Listing observations goes through the rate-limited Network API (authenticated +
throttled); downloading the waterfall PNGs hits S3 (not rate-limited). Held-out
stations are chosen from the data (smaller stations accumulated to ~20% of train
records) for the station-generalization test. The held-out SATELLITE (FrontierSat)
is the cross-satellite test.

Pushing to the Hub is a SEPARATE explicit step (pass --push, requires HF auth).

Setup:  set -a; source .env; set +a
Usage:  python scripts/build_and_push.py [cap_per_class] [--push]
"""
import dataclasses
import json
import os
import sys
from collections import Counter

import requests

from satnogs_signal.shared.satnogs_api import iter_observations, Observation
from satnogs_signal.data.build_dataset import build_records, to_dataset_dict, push, REPO_ID
from satnogs_signal.data.splits import SplitConfig

TOKEN = os.environ.get("satnogs_network_api_key") or None
THROTTLE = 1.0
OUT = "_dataset_build"  # gitignored local save dir
CACHE_DIR = os.path.join(OUT, "cache")  # per-(satellite,status) listing cache (resumable)

TRAIN = {63235: "OTP-2", 57175: "CUBEBEL-2", 68506: "AEPEX", 60246: "CatSat"}
HELDOUT_SAT = {69015: "FrontierSat"}

args = [a for a in sys.argv[1:] if not a.startswith("--")]
CAP = int(args[0]) if args else 120  # gold obs per class per satellite
DO_PUSH = "--push" in sys.argv

session = requests.Session()


def fetch_bytes(url: str) -> bytes:
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def collect(norad: int, name: str) -> list:
    """List gold obs for a satellite, caching each (norad, status) so a rate-limit
    cooldown only pauses progress — a re-run loads cached listings and resumes."""
    obs = []
    pages = (CAP + 24) // 25
    for status in ("with-signal", "without-signal"):
        cache_file = os.path.join(CACHE_DIR, f"{norad}_{status}.json")
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                got = [Observation(**d) for d in json.load(f)]
            print(f"  cached {len(got):>4} {status:<14} for {name} ({norad})", file=sys.stderr)
        else:
            got = list(
                iter_observations(
                    norad_cat_id=norad, waterfall_status=status, session=session,
                    max_pages=pages, token=TOKEN, request_interval=THROTTLE,
                    max_retries=8, backoff_base=2.0,
                )
            )[:CAP]
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump([dataclasses.asdict(o) for o in got], f)
            print(f"  listed {len(got):>4} {status:<14} for {name} ({norad})", file=sys.stderr)
        obs.extend(got)
    return obs


def choose_heldout_stations(train_records: list, frac: float = 0.20) -> set:
    counts = Counter(r["station"] for r in train_records)
    target = frac * len(train_records)
    held, covered = set(), 0
    for station, c in sorted(counts.items(), key=lambda kv: kv[1]):  # smaller first
        if covered >= target or len(held) >= len(counts) - 1:  # never hold out ALL stations
            break
        held.add(station)
        covered += c
    return held


def main() -> None:
    if not TOKEN:
        print("WARNING: no satnogs_network_api_key in env — listing will be rate-limited.",
              file=sys.stderr)

    all_obs = []
    for norad, name in {**TRAIN, **HELDOUT_SAT}.items():
        all_obs.extend(collect(norad, name))

    print(f"downloading + preprocessing {len(all_obs)} waterfalls (S3)...", file=sys.stderr)
    records = build_records(all_obs, fetch_bytes=fetch_bytes)
    print(f"  built {len(records)} records", file=sys.stderr)

    train_records = [r for r in records if r["norad"] in TRAIN]
    heldout_stations = choose_heldout_stations(train_records)

    cfg = SplitConfig(
        train_norads=list(TRAIN),
        heldout_satellite_norad=next(iter(HELDOUT_SAT)),
        heldout_station_ids=heldout_stations,
        val_fraction_by_time=0.2,
    )
    dd = to_dataset_dict(records, cfg)

    print(f"\nHeld-out stations ({len(heldout_stations)}): {sorted(heldout_stations)}")
    for split in ("train", "val", "test"):
        labels = Counter(dd[split]["label"])
        print(f"  {split:<5} {dd[split].num_rows:>5} rows  "
              f"(with-signal={labels.get(1, 0)}, without-signal={labels.get(0, 0)})")

    dd.save_to_disk(OUT)
    print(f"\nsaved DatasetDict to ./{OUT}/")

    if DO_PUSH:
        print(f"pushing to the Hub: {REPO_ID} (private)...")
        push(dd)
        print("pushed.")
    else:
        print("(not pushed — re-run with --push once HF auth is set up)")


if __name__ == "__main__":
    main()
