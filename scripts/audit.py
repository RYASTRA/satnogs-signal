"""Audit the live SatNOGS network for candidate satellites (build-time tool).

Samples gold-labelled observations across the network and ranks satellites by how
many *balanced* (both with-signal AND without-signal) gold labels they have, so you
can pick the narrow-and-deep training satellites + one held-out satellite.

Read-only and polite: authenticates with SATNOGS_API_TOKEN (higher rate limit) and
throttles to one page per THROTTLE_S. Put your token in a gitignored `.env` and run:

    set -a; source .env; set +a
    python scripts/audit.py [pages_per_class]
"""

import os
import sys

import requests

from satnogs_signal.shared.satnogs_api import FetchPolicy, iter_observations
from satnogs_signal.data.audit import tally_candidates

PAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 12  # 25 obs per page
THROTTLE_S = 1.0
# Accept the SatNOGS Network token under a few common env-var names.
TOKEN = (
    os.environ.get("satnogs_network_api_key")
    or os.environ.get("SATNOGS_NETWORK_API_KEY")
    or os.environ.get("SATNOGS_API_TOKEN")
    or None
)


def _sample(session) -> list:
    """Sample gold observations of both classes from the network (authenticated + throttled)."""
    policy = FetchPolicy(
        token=TOKEN, request_interval=THROTTLE_S, max_retries=8, backoff_base=2.0
    )
    sampled = []
    for status in ("with-signal", "without-signal"):
        n = 0
        for obs in iter_observations(
            waterfall_status=status, session=session, max_pages=PAGES, policy=policy
        ):
            sampled.append(obs)
            n += 1
        print(f"  sampled {n} {status} observations", file=sys.stderr)
    return sampled


def _fetch_names(session) -> dict:
    """Look up satellite names by NORAD id (best-effort; empty dict on failure)."""
    names = {}
    try:
        r = session.get(
            "https://db.satnogs.org/api/satellites/?format=json", timeout=90
        )
        for s in r.json():
            names[s.get("norad_cat_id")] = s.get("name")
    except (requests.RequestException, ValueError) as e:  # pragma: no cover
        print(f"  (satellite-name lookup failed: {e})", file=sys.stderr)
    return names


def main() -> None:
    """Sample gold observations, tally candidates, and print the ranked table."""
    session = requests.Session()
    sampled = _sample(session)
    tally = tally_candidates(sampled)
    names = _fetch_names(session)

    # Rank by the smaller of the two class counts (we need BOTH signal and no-signal).
    ranked = sorted(
        tally.items(),
        key=lambda kv: (
            min(kv[1]["with-signal"], kv[1]["without-signal"]),
            kv[1]["with-signal"] + kv[1]["without-signal"],
        ),
        reverse=True,
    )

    auth = "authenticated" if TOKEN else "ANONYMOUS (no token — low rate limit)"
    print(
        f"\n[{auth}] sampled {len(sampled)} gold observations across "
        f"{len(tally)} satellites ({PAGES} pages/class).\n"
    )
    print(f"{'NORAD':>7}  {'name':<26} {'+sig':>5} {'-sig':>5} {'min':>4}   modes")
    print("-" * 78)
    for norad, e in ranked[:25]:
        nm = (names.get(norad) or "?")[:26]
        mn = min(e["with-signal"], e["without-signal"])
        modes = ",".join(sorted(m for m in e["modes"] if m))[:24]
        print(
            f"{norad:>7}  {nm:<26} {e['with-signal']:>5} "
            f"{e['without-signal']:>5} {mn:>4}   {modes}"
        )


if __name__ == "__main__":
    main()
