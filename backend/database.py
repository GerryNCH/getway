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

            CREATE TABLE IF NOT EXISTS reviews (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id    TEXT NOT NULL,       -- itinerary video_id, or a
                                                  -- fixed key for static demo
                                                  -- routes (e.g. "mallorca-demo-route")
                name        TEXT NOT NULL,
                title       TEXT NOT NULL,
                rating      INTEGER NOT NULL,
                text        TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_reviews_video_id
                ON reviews (video_id);

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
        if "comments_json" not in existing_cols:
            conn.execute("ALTER TABLE itineraries ADD COLUMN comments_json TEXT DEFAULT '[]'")
        if "hero_attribution_json" not in existing_cols:
            conn.execute("ALTER TABLE itineraries ADD COLUMN hero_attribution_json TEXT DEFAULT NULL")
        if "gallery_attributions_json" not in existing_cols:
            conn.execute("ALTER TABLE itineraries ADD COLUMN gallery_attributions_json TEXT DEFAULT '[]'")
        if "status" not in existing_cols:
            # 'pending' | 'approved' | 'rejected'. Existing rows (generated
            # before the admin panel existed) default to 'approved' so they
            # keep working exactly as before — only newly-generated routes
            # start out pending review.
            conn.execute("ALTER TABLE itineraries ADD COLUMN status TEXT DEFAULT 'pending'")
            conn.execute("UPDATE itineraries SET status = 'approved' WHERE status IS NULL OR status = 'pending'")
        if "price_category" not in existing_cols:
            conn.execute("ALTER TABLE itineraries ADD COLUMN price_category TEXT DEFAULT ''")
        if "tags_json" not in existing_cols:
            conn.execute("ALTER TABLE itineraries ADD COLUMN tags_json TEXT DEFAULT '[]'")
        if "creator_handle" not in existing_cols:
            conn.execute("ALTER TABLE itineraries ADD COLUMN creator_handle TEXT DEFAULT ''")

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
    comments = json.loads(row["comments_json"] or "[]")
    hero_attribution = json.loads(row["hero_attribution_json"]) if row["hero_attribution_json"] else None
    gallery_attributions = json.loads(row["gallery_attributions_json"] or "[]")
    return Itinerary(
        destination=row["destination"],
        duration=row["duration"],
        days=days,
        hero_photo_url=row["hero_photo_url"] or "",
        hero_attribution=hero_attribution,
        gallery_photo_urls=gallery_urls,
        gallery_attributions=gallery_attributions,
        comments=comments,
    )


def _attr_to_dict(attr) -> dict | None:
    """
    Normalizes an attribution value to a plain dict for JSON storage.
    Accepts either an UnsplashAttribution instance (has .model_dump()) or
    an already-plain dict (places.py sets it directly as a dict, which
    doesn't have .model_dump()) — calling .model_dump() unconditionally
    crashed on the latter.
    """
    if attr is None:
        return None
    return attr.model_dump() if hasattr(attr, "model_dump") else dict(attr)


def save_itinerary(video_id: str, url: str, itinerary: Itinerary) -> None:
    """Saves a freshly extracted itinerary to the cache."""
    with _conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO itineraries
               (video_id, url, destination, duration, days_json, created_at,
                hero_photo_url, gallery_photo_urls_json, comments_json,
                hero_attribution_json, gallery_attributions_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                video_id,
                url,
                itinerary.destination,
                itinerary.duration,
                json.dumps([d.model_dump() for d in itinerary.days]),
                datetime.utcnow().isoformat(),
                itinerary.hero_photo_url,
                json.dumps(itinerary.gallery_photo_urls),
                json.dumps([c.model_dump() for c in itinerary.comments]),
                json.dumps(_attr_to_dict(itinerary.hero_attribution)),
                json.dumps([_attr_to_dict(a) for a in itinerary.gallery_attributions]),
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


def _row_to_admin_dict(row: sqlite3.Row) -> dict:
    """
    Full itinerary content + admin metadata (status, video_id, url,
    created_at) — everything the admin panel needs to render a preview
    and inline editor without a second request.
    """
    return {
        "video_id": row["video_id"],
        "url": row["url"],
        "destination": row["destination"],
        "duration": row["duration"],
        "days": json.loads(row["days_json"]),
        "hero_photo_url": row["hero_photo_url"] or "",
        "gallery_photo_urls": json.loads(row["gallery_photo_urls_json"] or "[]"),
        "status": row["status"] or "pending",
        "created_at": row["created_at"],
        "price_category": row["price_category"] or "",
        "tags": json.loads(row["tags_json"] or "[]"),
        "creator_handle": row["creator_handle"] or "",
    }


def set_route_meta(video_id: str, price_category: str, tags: list[str], creator_handle: str) -> bool:
    """
    Saves the homepage-grid curation fields an admin sets when approving a
    route (price category, filter tags, creator handle) — separate from
    update_itinerary_content since these aren't part of the Itinerary
    content model itself. Returns False if video_id doesn't exist.
    """
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE itineraries SET price_category = ?, tags_json = ?, creator_handle = ? WHERE video_id = ?",
            (price_category, json.dumps(tags), creator_handle, video_id),
        )
    return cur.rowcount > 0


def list_public_approved() -> list[dict]:
    """
    Lightweight summary of every approved route — everything the homepage
    route grid needs to render a card, and nothing more (no full stop
    content, no admin-only fields). Used by the public GET /routes endpoint.
    """
    with _conn() as conn:
        rows = conn.execute(
            """SELECT video_id, destination, duration, days_json, hero_photo_url,
                      price_category, tags_json, creator_handle
               FROM itineraries WHERE status = 'approved' ORDER BY created_at DESC"""
        ).fetchall()
    result = []
    for r in rows:
        days = json.loads(r["days_json"])
        stop_count = sum(len(d.get("stops", [])) for d in days)
        result.append({
            "video_id": r["video_id"],
            "destination": r["destination"],
            "duration": r["duration"],
            "day_count": len(days),
            "stop_count": stop_count,
            "hero_photo_url": r["hero_photo_url"] or "",
            "price_category": r["price_category"] or "€€",
            "tags": json.loads(r["tags_json"] or "[]"),
            "creator_handle": r["creator_handle"] or "",
        })
    return result


def list_by_status(status: str) -> list[dict]:
    """Returns full itinerary content for every route with the given status."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM itineraries WHERE status = ? ORDER BY created_at DESC",
            (status,),
        ).fetchall()
    return [_row_to_admin_dict(r) for r in rows]


def set_status(video_id: str, status: str) -> bool:
    """Updates just the status column. Returns False if video_id doesn't exist."""
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE itineraries SET status = ? WHERE video_id = ?", (status, video_id)
        )
    return cur.rowcount > 0


def delete_itinerary_permanently(video_id: str) -> bool:
    """Hard-deletes a route. Used by the admin 'Изтрий' button (not 'Отхвърли')."""
    with _conn() as conn:
        cur = conn.execute("DELETE FROM itineraries WHERE video_id = ?", (video_id,))
    return cur.rowcount > 0


def update_itinerary_content(video_id: str, itinerary: Itinerary) -> bool:
    """
    Overwrites the editable content of a route (days/stops, hero photo,
    gallery, destination, duration) from the admin inline editor.
    Deliberately does NOT touch status/url/created_at/comments — those
    aren't part of what the admin editor edits.
    Returns False if video_id doesn't exist.
    """
    with _conn() as conn:
        cur = conn.execute(
            """UPDATE itineraries
               SET destination = ?, duration = ?, days_json = ?,
                   hero_photo_url = ?, gallery_photo_urls_json = ?
               WHERE video_id = ?""",
            (
                itinerary.destination,
                itinerary.duration,
                json.dumps([d.model_dump() for d in itinerary.days]),
                itinerary.hero_photo_url,
                json.dumps(itinerary.gallery_photo_urls),
                video_id,
            ),
        )
    return cur.rowcount > 0


def get_stats() -> dict:
    """Counts for the admin Statistics tab: totals by status + top destinations."""
    with _conn() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM itineraries").fetchone()["c"]
        pending = conn.execute(
            "SELECT COUNT(*) c FROM itineraries WHERE status = 'pending'"
        ).fetchone()["c"]
        approved = conn.execute(
            "SELECT COUNT(*) c FROM itineraries WHERE status = 'approved'"
        ).fetchone()["c"]
        rejected = conn.execute(
            "SELECT COUNT(*) c FROM itineraries WHERE status = 'rejected'"
        ).fetchone()["c"]
        top_rows = conn.execute(
            """SELECT destination, COUNT(*) c FROM itineraries
               GROUP BY destination ORDER BY c DESC LIMIT 5"""
        ).fetchall()
    return {
        "total": total,
        "pending": pending,
        "approved": approved,
        "rejected": rejected,
        "top_destinations": [{"destination": r["destination"], "count": r["c"]} for r in top_rows],
    }


def clear_all_itineraries() -> int:
    """
    Deletes every cached itinerary. Used by the admin 'Clear Cache' button
    so previously-generated routes regenerate fresh (e.g. after a link
    format or pricing change) instead of serving stale cached data.
    Returns the number of rows deleted.
    """
    with _conn() as conn:
        cur = conn.execute("DELETE FROM itineraries")
        count = cur.rowcount
    print(f"[DB] Cleared {count} cached itineraries")
    return count


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


# ── Reviews ────────────────────────────────────────────────────────────────

def save_review(video_id: str, name: str, title: str, rating: int, text: str) -> dict:
    """Saves a new review and returns it as a dict (ready for Review(**dict))."""
    created_at = datetime.utcnow().isoformat()
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO reviews (video_id, name, title, rating, text, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (video_id, name, title, rating, text, created_at),
        )
        new_id = cur.lastrowid
    print(f"[DB] Saved review #{new_id} for {video_id} ({rating}★)")
    return {
        "id": new_id,
        "video_id": video_id,
        "name": name,
        "title": title,
        "rating": rating,
        "text": text,
        "created_at": created_at,
    }


def get_reviews(video_id: str) -> list[dict]:
    """Returns all reviews for a video_id, most recent first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM reviews WHERE video_id = ? ORDER BY created_at DESC",
            (video_id,),
        ).fetchall()
    return [dict(r) for r in rows]
