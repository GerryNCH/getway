"""
database.py — SQLite cache layer.

Two tables:
  itineraries  — full extracted routes, keyed by video_id
  troll_cache  — previous troll-filter decisions (avoid re-checking same URL)

SQLite is perfect for this stage: zero setup, single file, fast reads.
Upgrade path: swap engine URL for PostgreSQL when you scale.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from models import Itinerary

DB_PATH = Path(__file__).parent / "getway.db"


def _conn() -> sqlite3.Connection:
    """Returns a thread-safe SQLite connection with dict rows."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Creates tables if they don't exist yet. Call once at app startup."""
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS itineraries (
                video_id    TEXT PRIMARY KEY,
                url         TEXT NOT NULL,
                destination TEXT NOT NULL,
                duration    TEXT NOT NULL,
                days_json   TEXT NOT NULL,       -- full Itinerary.days as JSON
                created_at  TEXT NOT NULL,
                added_by    TEXT DEFAULT 'ai'    -- 'ai' | 'manual' (admin panel later)
            );

            CREATE TABLE IF NOT EXISTS troll_cache (
                video_id    TEXT PRIMARY KEY,
                is_travel   INTEGER NOT NULL,    -- 1 = travel, 0 = rejected
                reason      TEXT,
                checked_at  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_itineraries_destination
                ON itineraries (destination);
        """)

        # Migration: earlier versions of this table didn't store the hero
        # image or gallery photos, so cached/shared routes lost them on
        # reload even though a fresh generation had them. Add the columns
        # if they're missing (safe to run every startup).
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(itineraries)")}
        if "hero_photo_url" not in existing_cols:
            conn.execute("ALTER TABLE itineraries ADD COLUMN hero_photo_url TEXT DEFAULT ''")
        if "gallery_photo_urls_json" not in existing_cols:
            conn.execute("ALTER TABLE itineraries ADD COLUMN gallery_photo_urls_json TEXT DEFAULT '[]'")

    print(f"[DB] Initialised at {DB_PATH}")


# ── Itinerary cache ───────────────────────────────────────────────────────────

def get_itinerary(video_id: str) -> Itinerary | None:
    """Returns a cached Itinerary or None if not found."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM itineraries WHERE video_id = ?", (video_id,)
        ).fetchone()
    if not row:
        return None
    days = json.loads(row["days_json"])
    gallery_urls = json.loads(row["gallery_photo_urls_json"] or "[]")
    return Itinerary(
        destination=row["destination"],
        duration=row["duration"],
        days=days,
        hero_photo_url=row["hero_photo_url"] or "",
        gallery_photo_urls=gallery_urls,
    )


def save_itinerary(video_id: str, url: str, itinerary: Itinerary) -> None:
    """Saves a freshly extracted itinerary to the cache."""
    with _conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO itineraries
               (video_id, url, destination, duration, days_json, created_at,
                hero_photo_url, gallery_photo_urls_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                video_id,
                url,
                itinerary.destination,
                itinerary.duration,
                json.dumps([d.model_dump() for d in itinerary.days]),
                datetime.utcnow().isoformat(),
                itinerary.hero_photo_url,
                json.dumps(itinerary.gallery_photo_urls),
            ),
        )
    print(f"[DB] Saved itinerary for {video_id} ({itinerary.destination})")


def list_itineraries() -> list[dict]:
    """Returns all cached itineraries (for admin panel later)."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT video_id, url, destination, duration, created_at, added_by "
            "FROM itineraries ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# ── Troll filter cache ────────────────────────────────────────────────────────

def get_troll_decision(video_id: str) -> bool | None:
    """
    Returns:
      True  — previously confirmed as travel content
      False — previously rejected as non-travel
      None  — never checked before
    """
    with _conn() as conn:
        row = conn.execute(
            "SELECT is_travel FROM troll_cache WHERE video_id = ?", (video_id,)
        ).fetchone()
    if row is None:
        return None
    return bool(row["is_travel"])


def save_troll_decision(video_id: str, is_travel: bool, reason: str) -> None:
    """Stores the result of a troll-filter check so we never repeat it."""
    with _conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO troll_cache
               (video_id, is_travel, reason, checked_at)
               VALUES (?, ?, ?, ?)""",
            (video_id, int(is_travel), reason, datetime.utcnow().isoformat()),
        )
