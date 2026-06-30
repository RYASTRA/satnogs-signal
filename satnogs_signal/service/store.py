"""Idempotent SQLite store of triage predictions, keyed by obs_id."""

from __future__ import annotations

import sqlite3
from typing import Iterable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    obs_id          INTEGER PRIMARY KEY,
    norad           INTEGER,
    mode            TEXT,
    station         INTEGER,
    timestamp       TEXT,
    waterfall_url   TEXT,
    p_signal        REAL,
    predicted_label INTEGER,
    scored_at       TEXT
);
"""


def connect(path: str) -> sqlite3.Connection:
    """Open the SQLite store at path, set row factory, and ensure the schema."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def upsert_prediction(conn: sqlite3.Connection, rec: dict) -> None:
    """Insert rec or, on obs_id conflict, update its score fields; commit."""
    conn.execute(
        """
        INSERT INTO predictions
            (obs_id, norad, mode, station, timestamp, waterfall_url,
             p_signal, predicted_label, scored_at)
        VALUES
            (:obs_id, :norad, :mode, :station, :timestamp, :waterfall_url,
             :p_signal, :predicted_label, :scored_at)
        ON CONFLICT(obs_id) DO UPDATE SET
            p_signal=excluded.p_signal,
            predicted_label=excluded.predicted_label,
            scored_at=excluded.scored_at
        """,
        rec,
    )
    conn.commit()


def already_scored(conn: sqlite3.Connection, obs_ids: Iterable[int]) -> set:
    """Return the subset of obs_ids that already have a stored prediction."""
    ids = list(obs_ids)
    if not ids:
        return set()
    placeholders = ",".join("?" * len(ids))
    q = f"SELECT obs_id FROM predictions WHERE obs_id IN ({placeholders})"
    return {r["obs_id"] for r in conn.execute(q, ids)}


def ranked(conn: sqlite3.Connection, limit: int = 100) -> list:
    """Return up to limit predictions as dicts, highest p_signal first."""
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY p_signal DESC, obs_id DESC LIMIT ?",
        (limit,),
    )
    return [dict(r) for r in rows]


def stats(conn: sqlite3.Connection) -> dict:
    """Return store-wide counts: total rows, mean p_signal, and signal count."""
    row = conn.execute(
        "SELECT COUNT(*) AS n, AVG(p_signal) AS avg_p, "
        "SUM(predicted_label) AS n_sig FROM predictions"
    ).fetchone()
    return {
        "n": row["n"],
        "avg_p_signal": row["avg_p"] if row["avg_p"] is not None else 0.0,
        "n_predicted_signal": row["n_sig"] or 0,
    }
