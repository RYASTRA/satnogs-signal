"""Poll the chosen satellites' unvetted observations and score them into triage.db.
Read-only. Setup: set the SatNOGS token in .env. Usage:
    docker compose run --rm app python scripts/run_poller.py [max_pages]
"""

import os
import sys

import requests

from satnogs_signal.model.train import load_scorer
from satnogs_signal.service import store, poller

MODEL = "ryroeu/satnogs-signal-classifier"
DB = "triage.db"
NORADS = [63235, 57175, 68506, 60246, 69015]
TOKEN = os.environ.get("satnogs_network_api_key") or None

_session = requests.Session()


def fetch_bytes(url: str) -> bytes:
    r = _session.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def main() -> None:
    max_pages = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"loading model {MODEL} ...", flush=True)
    score = load_scorer(MODEL)  # load the model ONCE; the poller reuses this closure
    conn = store.connect(DB)
    n = poller.poll(
        NORADS,
        conn,
        score_fn=score,
        fetch_bytes=fetch_bytes,
        token=TOKEN,
        max_pages=max_pages,
    )
    print(f"scored {n} new observations into {DB}")
    print(store.stats(conn))


if __name__ == "__main__":
    main()
