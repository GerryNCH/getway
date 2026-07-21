"""
database.py — SQLite cache layer.

Two tables:
  itineraries  — full extracted routes, keyed by video_id
  troll_cache  — previous troll-filter decisions (avoid re-checking same URL)

SQLite is perfect for this stage: zero setup, single file, fast reads.
Upgrade path: swap engine URL for PostgreSQL when you scale.
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from models import Itinerary

# CRITICAL: this must point at a Railway Volume mount (persistent disk),
# NOT a path inside the code directory. The code directory gets rebuilt
# from scratch on every deploy — a database file living there is wiped
# every single time new code is pushed. Set the DATA_DIR environment
# variable to the Volume's mount path (e.g. "/data") in Railway →
# Variables. Falls back to the old (non-persistent!) behavior only if
# DATA_DIR isn't set, so local development still works without a volume.
DATA_DIR = Path(os.getenv("DATA_DIR", str(Path(__file__).parent)))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "getway.db"


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

            CREATE TABLE IF NOT EXISTS site_settings (
                id                       INTEGER PRIMARY KEY CHECK (id = 1),
                hero_slides_json         TEXT DEFAULT '[]',
                featured_route_ids_json  TEXT DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS stop_cache (
                cache_key          TEXT PRIMARY KEY,
                city               TEXT,
                stop_name          TEXT,
                photo_url          TEXT,
                maps_url_override  TEXT DEFAULT '',
                updated_at         TEXT
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
        if "summary" not in existing_cols:
            conn.execute("ALTER TABLE itineraries ADD COLUMN summary TEXT DEFAULT ''")
        if "generation_cost_usd" not in existing_cols:
            conn.execute("ALTER TABLE itineraries ADD COLUMN generation_cost_usd REAL DEFAULT 0.0")
        if "view_count" not in existing_cols:
            conn.execute("ALTER TABLE itineraries ADD COLUMN view_count INTEGER DEFAULT 0")
        if "affiliate_click_count" not in existing_cols:
            conn.execute("ALTER TABLE itineraries ADD COLUMN affiliate_click_count INTEGER DEFAULT 0")
        if "hotel_banner_photo_url" not in existing_cols:
            conn.execute("ALTER TABLE itineraries ADD COLUMN hotel_banner_photo_url TEXT DEFAULT ''")
        if "car_rental_recommended" not in existing_cols:
            conn.execute("ALTER TABLE itineraries ADD COLUMN car_rental_recommended INTEGER DEFAULT 0")
        if "car_rental_note" not in existing_cols:
            conn.execute("ALTER TABLE itineraries ADD COLUMN car_rental_note TEXT DEFAULT ''")

        # Seed the singleton site_settings row once, with the hero slides
        # that were previously hardcoded in index.html — so nothing changes
        # visually on the homepage until an admin actually edits them.
        row = conn.execute("SELECT id FROM site_settings WHERE id = 1").fetchone()
        if row is None:
            default_hero_slides = json.dumps([
                "https://images.unsplash.com/photo-1476514525535-07fb3b4ae5f1?w=2000&auto=format&fit=crop",
                "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=2000&auto=format&fit=crop",
                "https://images.unsplash.com/photo-1488085061387-422e29b40080?w=2000&auto=format&fit=crop",
            ])
            conn.execute(
                "INSERT INTO site_settings (id, hero_slides_json, featured_route_ids_json) VALUES (1, ?, '[]')",
                (default_hero_slides,),
            )

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
        summary=row["summary"] or "",
        creator_handle=row["creator_handle"] or "",
        price_category=row["price_category"] or "",
        generation_cost_usd=row["generation_cost_usd"] or 0.0,
        hotel_banner_photo_url=row["hotel_banner_photo_url"] or "",
        car_rental_recommended=bool(row["car_rental_recommended"]),
        car_rental_note=row["car_rental_note"] or "",
        view_count=row["view_count"] or 0,
        affiliate_click_count=row["affiliate_click_count"] or 0,
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
                hero_attribution_json, gallery_attributions_json, summary,
                generation_cost_usd, car_rental_recommended, car_rental_note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                itinerary.summary,
                itinerary.generation_cost_usd,
                int(itinerary.car_rental_recommended),
                itinerary.car_rental_note,
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
        "summary": row["summary"] or "",
        "hero_photo_url": row["hero_photo_url"] or "",
        "gallery_photo_urls": json.loads(row["gallery_photo_urls_json"] or "[]"),
        "status": row["status"] or "pending",
        "created_at": row["created_at"],
        "price_category": row["price_category"] or "",
        "tags": json.loads(row["tags_json"] or "[]"),
        "creator_handle": row["creator_handle"] or "",
        "generation_cost_usd": row["generation_cost_usd"] or 0.0,
        "view_count": row["view_count"] or 0,
        "affiliate_click_count": row["affiliate_click_count"] or 0,
        "hotel_banner_photo_url": row["hotel_banner_photo_url"] or "",
        "car_rental_recommended": bool(row["car_rental_recommended"]),
        "car_rental_note": row["car_rental_note"] or "",
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
                      price_category, tags_json, creator_handle, summary
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
            "summary": r["summary"] or "",
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
                   hero_photo_url = ?, gallery_photo_urls_json = ?, summary = ?,
                   hotel_banner_photo_url = ?, car_rental_recommended = ?,
                   car_rental_note = ?
               WHERE video_id = ?""",
            (
                itinerary.destination,
                itinerary.duration,
                json.dumps([d.model_dump() for d in itinerary.days]),
                itinerary.hero_photo_url,
                json.dumps(itinerary.gallery_photo_urls),
                itinerary.summary,
                itinerary.hotel_banner_photo_url,
                int(itinerary.car_rental_recommended),
                itinerary.car_rental_note,
                video_id,
            ),
        )
    return cur.rowcount > 0


def get_stats() -> dict:
    """Counts for the admin Statistics tab: totals by status, top destinations, cost, and engagement."""
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
        total_cost = conn.execute(
            "SELECT COALESCE(SUM(generation_cost_usd), 0) c FROM itineraries"
        ).fetchone()["c"]
        total_views = conn.execute(
            "SELECT COALESCE(SUM(view_count), 0) c FROM itineraries"
        ).fetchone()["c"]
        total_affiliate_clicks = conn.execute(
            "SELECT COALESCE(SUM(affiliate_click_count), 0) c FROM itineraries"
        ).fetchone()["c"]
        most_viewed_row = conn.execute(
            """SELECT video_id, destination, view_count FROM itineraries
               WHERE view_count > 0 ORDER BY view_count DESC LIMIT 1"""
        ).fetchone()
        most_clicked_row = conn.execute(
            """SELECT video_id, destination, affiliate_click_count FROM itineraries
               WHERE affiliate_click_count > 0 ORDER BY affiliate_click_count DESC LIMIT 1"""
        ).fetchone()
    return {
        "total": total,
        "pending": pending,
        "approved": approved,
        "rejected": rejected,
        "top_destinations": [{"destination": r["destination"], "count": r["c"]} for r in top_rows],
        "total_generation_cost_usd": total_cost,
        "total_views": total_views,
        "total_affiliate_clicks": total_affiliate_clicks,
        "most_viewed": (
            {"destination": most_viewed_row["destination"], "views": most_viewed_row["view_count"]}
            if most_viewed_row else None
        ),
        "most_clicked": (
            {"destination": most_clicked_row["destination"], "clicks": most_clicked_row["affiliate_click_count"]}
            if most_clicked_row else None
        ),
    }


def increment_view_count(video_id: str) -> bool:
    """Bumps a route's view counter by 1. Silently no-ops if the video_id doesn't exist."""
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE itineraries SET view_count = view_count + 1 WHERE video_id = ?", (video_id,)
        )
    return cur.rowcount > 0


def increment_affiliate_click_count(video_id: str) -> bool:
    """Bumps a route's affiliate-link-click counter by 1 (Booking/Expedia/Airbnb buttons)."""
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE itineraries SET affiliate_click_count = affiliate_click_count + 1 WHERE video_id = ?",
            (video_id,),
        )
    return cur.rowcount > 0


def get_site_settings() -> dict:
    """
    Returns the homepage's admin-controlled settings: hero_slides (list of
    image URLs for the rotating homepage background) and featured_route_ids
    (ordered list of video_ids to show on the homepage grid — empty means
    "show all approved routes automatically", the original default behavior).
    """
    with _conn() as conn:
        row = conn.execute("SELECT * FROM site_settings WHERE id = 1").fetchone()
    if row is None:
        return {"hero_slides": [], "featured_route_ids": []}
    return {
        "hero_slides": json.loads(row["hero_slides_json"] or "[]"),
        "featured_route_ids": json.loads(row["featured_route_ids_json"] or "[]"),
    }


def set_site_settings(hero_slides: list[str], featured_route_ids: list[str]) -> None:
    """Overwrites the homepage settings row (upsert — creates it if somehow missing)."""
    with _conn() as conn:
        conn.execute(
            """INSERT INTO site_settings (id, hero_slides_json, featured_route_ids_json)
               VALUES (1, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 hero_slides_json = excluded.hero_slides_json,
                 featured_route_ids_json = excluded.featured_route_ids_json""",
            (json.dumps(hero_slides), json.dumps(featured_route_ids)),
        )


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


# ── Troll-filter decision cache ──────────────────────────────────────────
# Avoids re-running the Haiku travel-content check (small but non-zero
# cost) for a video_id that's already been checked before — e.g. a repeat
# submission of the same link, or a re-extraction after Clear cache.

def get_troll_decision(video_id: str) -> bool | None:
    """Returns the cached is_travel decision for this video_id, or None if never checked."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT is_travel FROM troll_cache WHERE video_id = ?", (video_id,)
        ).fetchone()
    if row is None:
        return None
    return bool(row["is_travel"])


def save_troll_decision(video_id: str, is_travel: bool, reason: str) -> None:
    """Saves (or updates) the troll-filter decision for a video_id."""
    with _conn() as conn:
        conn.execute(
            """INSERT INTO troll_cache (video_id, is_travel, reason, checked_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(video_id) DO UPDATE SET
                   is_travel = excluded.is_travel,
                   reason = excluded.reason,
                   checked_at = excluded.checked_at""",
            (video_id, int(is_travel), reason, datetime.utcnow().isoformat()),
        )


def get_reviews(video_id: str) -> list[dict]:
    """Returns all reviews for a video_id, most recent first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM reviews WHERE video_id = ? ORDER BY created_at DESC",
            (video_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Stop photo/address cache — reuse across routes for the same city ────────
# When the same destination gets a second (or third) route generated later,
# stops that already appeared before (e.g. "Colosseum" showing up again in a
# second Rome route) shouldn't have to be re-fetched and re-fixed by hand
# every time — reuse whatever photo and address Gerry already approved or
# fixed for that exact stop in this city.

def _stop_cache_key(city: str, stop_name: str) -> str:
    """Normalizes city+name into a stable lookup key (lowercase, trimmed)."""
    return f"{(city or '').strip().lower()}|{(stop_name or '').strip().lower()}"


def get_cached_stop(city: str, stop_name: str) -> dict | None:
    """
    Returns {"photo_url": ..., "maps_url_override": ...} if this exact
    city+stop_name combination was already saved from a previous route,
    else None. Called during photo enrichment, before falling back to a
    fresh Places/Unsplash lookup.
    """
    key = _stop_cache_key(city, stop_name)
    with _conn() as conn:
        row = conn.execute(
            "SELECT photo_url, maps_url_override FROM stop_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
    if not row or not row["photo_url"]:
        return None
    return {"photo_url": row["photo_url"], "maps_url_override": row["maps_url_override"] or ""}


def upsert_stop_cache(city: str, stop_name: str, photo_url: str, maps_url_override: str = "") -> None:
    """Saves/updates one stop's photo+address for reuse in future routes for the same city."""
    if not stop_name or not photo_url:
        return
    key = _stop_cache_key(city, stop_name)
    with _conn() as conn:
        conn.execute(
            """INSERT INTO stop_cache (cache_key, city, stop_name, photo_url, maps_url_override, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(cache_key) DO UPDATE SET
                   photo_url = excluded.photo_url,
                   maps_url_override = excluded.maps_url_override,
                   updated_at = excluded.updated_at""",
            (key, city, stop_name, photo_url, maps_url_override or "", datetime.utcnow().isoformat()),
        )


def cache_stops_from_itinerary(destination: str, itinerary: Itinerary) -> None:
    """
    Bulk-saves every stop's current photo+address into the cache — called
    whenever an admin saves edits or approves a route, since that's the
    signal that the photos in it (at least the ones worth keeping) are
    good. Cheap no-op for stops with no photo set.
    """
    city = (destination or "").split(",")[0].strip()
    for day in itinerary.days:
        for stop in day.stops:
            upsert_stop_cache(city, stop.name, stop.photo_url, getattr(stop, "maps_url_override", ""))
