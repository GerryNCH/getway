"""
main.py — FastAPI entry point.

The full 5-layer pipeline per request:
  1. Validate URL
  2. Extract stable video_id
  3. Database check → serve from cache if hit (free, instant)
  4. Fetch metadata (fast, no download)
  5. Troll filter via Claude Haiku (cheap: ~$0.0003)
  6. Download video + extract frames
  7. Claude Sonnet multimodal analysis (~$0.02-0.05)
  8. Save to database
  9. Return itinerary

Run locally:
  uvicorn main:app --reload --port 8000
"""

import tempfile

# Load .env FIRST — before any module that reads ANTHROPIC_API_KEY
from dotenv import load_dotenv
load_dotenv()

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import ExtractRequest, ExtractResponse, Itinerary, Comment, ReviewCreate, Review, ReviewsResponse, RouteMeta, SiteSettings
import database
from extractor import (
    extract_video_id, fetch_metadata, download_video, extract_frames,
    is_slideshow, fetch_slideshow_post, download_slideshow_images,
    fetch_top_comments, is_instagram_url, fetch_instagram_post,
    download_instagram_video,
)
from troll_filter import check_is_travel
from ai_analyzer import analyse_frames
from places import enrich_itinerary_with_photos, _unsplash_candidates, _attribution_from_candidate, _trigger_unsplash_download

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="GetWay Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://getway-theta.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "*",            # tighten this before production launch
    ],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    database.init_db()
    print("[Startup] GetWay backend ready")


# ── Main extraction endpoint ──────────────────────────────────────────────────

@app.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest):
    url = req.url.strip()
    if not url.startswith("http"):
        raise HTTPException(400, "Invalid URL — must start with http or https")

    # ── Layer 1: stable video ID ──────────────────────────────────────────────
    video_id = extract_video_id(url)
    print(f"\n[Request] {video_id}  {url}")

    # ── Layer 2: database cache check (FREE) ──────────────────────────────────
    cached = database.get_itinerary(video_id)
    if cached:
        print(f"[Cache HIT] Serving {video_id} from database — $0 spent")
        return ExtractResponse(
            itinerary=cached,
            source="cache",
            video_id=video_id,
            cached=True,
        )
    print(f"[Cache MISS] {video_id} not in database — proceeding to AI pipeline")

    # ── Layer 3: fetch metadata (fast, no download) ───────────────────────────
    # Slideshow (/photo/) posts can't go through yt-dlp at all — it has no
    # /photo/ support (confirmed: not in its URL regex, and the feature
    # request was closed upstream as "wontfix"). For those, one Apify call
    # gets us both the caption (for the troll filter below) and the slide
    # image URLs, so we stash the full result to reuse after the filter.
    # Instagram Reels similarly can't go through yt-dlp reliably (Instagram
    # aggressively blocks datacenter IPs) — one Apify call gets us the
    # caption, the direct video URL, AND the top comments all at once, so
    # that's stashed too and reused below (skipping a second comments call).
    slideshow_data = None
    instagram_data = None
    if is_slideshow(url):
        try:
            slideshow_data = fetch_slideshow_post(url)
            meta = {"title": slideshow_data["title"], "description": slideshow_data["description"]}
            print(f"[Meta] (slideshow) Title: {meta['title'][:60]}")
        except RuntimeError as e:
            raise HTTPException(422, f"Could not fetch slideshow info: {e}")
    elif is_instagram_url(url):
        try:
            instagram_data = fetch_instagram_post(url)
            meta = {
                "title": instagram_data["title"],
                "description": instagram_data["description"],
                "uploader": instagram_data["uploader"],
            }
            print(f"[Meta] (Instagram) Title: {meta['title'][:60]}")
        except RuntimeError as e:
            raise HTTPException(422, f"Could not fetch Instagram post info: {e}")
    else:
        try:
            meta = fetch_metadata(url)
            print(f"[Meta] Title: {meta['title'][:60]}")
        except RuntimeError as e:
            raise HTTPException(422, f"Could not fetch video info: {e}")

    # ── Layer 4: troll filter — Claude Haiku (~$0.0003) ───────────────────────
    is_travel, reason, troll_cost_usd = check_is_travel(
        video_id,
        meta["title"],
        meta["description"],
    )
    print(f"[TrollFilter] is_travel={is_travel}  reason={reason}")

    if not is_travel:
        raise HTTPException(
            422,
            f"This video doesn't appear to be travel content ({reason}). "
            "Please paste a link to a travel vlog or destination video."
        )

    # ── Layers 5–7: download → frames → multimodal AI ────────────────────────
    with tempfile.TemporaryDirectory() as tmp:

        if is_slideshow(url):
            # Images URLs were already fetched above — just download them.
            try:
                frames = download_slideshow_images(slideshow_data["image_urls"], tmp)
                print(f"[Slideshow] Downloaded {len(frames)} slide images")
            except RuntimeError as e:
                raise HTTPException(422, f"Slideshow download failed: {e}")
        elif is_instagram_url(url):
            # Video URL was already resolved above by the same Apify call.
            try:
                video_path = download_instagram_video(instagram_data["video_url"], tmp)
                print(f"[Instagram] Downloaded video → {video_path}")
            except RuntimeError as e:
                raise HTTPException(422, f"Instagram video download failed: {e}")

            try:
                frames = extract_frames(video_path, tmp, req.max_frames)
                print(f"[Frames] Extracted {len(frames)} frames")
            except RuntimeError as e:
                raise HTTPException(500, f"Frame extraction failed: {e}")
        else:
            # Regular video post — download then extract evenly-spaced frames
            try:
                video_path = download_video(url, tmp)
                print(f"[Download] {video_path}")
            except RuntimeError as e:
                raise HTTPException(422, f"Video download failed: {e}")

            try:
                frames = extract_frames(video_path, tmp, req.max_frames)
                print(f"[Frames] Extracted {len(frames)} frames")
            except RuntimeError as e:
                raise HTTPException(500, f"Frame extraction failed: {e}")

        # ── Layer 7a: fetch real comments early (non-fatal) ───────────────────
        # Fetched BEFORE the AI analysis (not after) so they can be used as
        # an identification aid — see analyse_frames' docstring. Instagram's
        # comments were already fetched in the same Apify call above, so
        # reuse those instead of an unnecessary second network call.
        raw_comments: list[dict] = []
        if instagram_data is not None:
            raw_comments = instagram_data.get("comments", [])
            print(f"[Comments] Using {len(raw_comments)} Instagram comments (already fetched)")
        else:
            try:
                raw_comments = fetch_top_comments(url, max_comments=15)
                print(f"[Comments] Fetched {len(raw_comments)} real TikTok comments")
            except Exception as e:
                print(f"[Comments] Fetch failed (non-fatal): {e}")

        # Claude multimodal analysis
        try:
            itinerary, ai_price_category, ai_tags, ai_cost_usd = analyse_frames(frames, comments=raw_comments)
            itinerary.generation_cost_usd = troll_cost_usd + ai_cost_usd
            print(f"[Cost] ${itinerary.generation_cost_usd:.4f} "
                  f"(troll ${troll_cost_usd:.4f} + analysis ${ai_cost_usd:.4f})")
            print(f"[AI] Destination: {itinerary.destination} — "
                  f"{sum(len(d.stops) for d in itinerary.days)} stops across "
                  f"{len(itinerary.days)} days")
        except Exception as e:
            raise HTTPException(500, f"AI analysis failed: {e}")

        # ── Layer 7b: enrich with Google Places photos ────────────────────────
        try:
            enrich_itinerary_with_photos(itinerary)
        except Exception as e:
            print(f"[Places] Enrichment failed (non-fatal): {e}")

        itinerary.comments = [Comment(**c) for c in raw_comments]

    # ── Layer 8: save to database ─────────────────────────────────────────────
    database.save_itinerary(video_id, url, itinerary)

    # Homepage-grid curation fields: price/tags come from the AI's own
    # estimate above; creator_handle comes from yt-dlp's "uploader" field
    # (regular videos only — the Apify slideshow path doesn't return a
    # confirmed author field, so it's left blank for admins to fill in).
    creator_handle = meta.get("uploader", "")
    if creator_handle and not creator_handle.startswith("@"):
        creator_handle = f"@{creator_handle}"
    database.set_route_meta(video_id, ai_price_category, ai_tags, creator_handle)
    itinerary.creator_handle = creator_handle
    itinerary.price_category = ai_price_category

    return ExtractResponse(
        itinerary=itinerary,
        source="ai_generated",
        video_id=video_id,
        cached=False,
    )


# ── Reviews ────────────────────────────────────────────────────────────────

@app.post("/reviews", response_model=Review)
def create_review(review: ReviewCreate):
    """Saves a review left by a traveler for a specific route (video_id)."""
    name = review.name.strip()
    title = review.title.strip()
    text = review.text.strip()
    video_id = review.video_id.strip()

    if not video_id:
        raise HTTPException(400, "Missing video_id")
    if not name or not title or not text:
        raise HTTPException(400, "Name, title, and review text are required")
    if not (1 <= review.rating <= 5):
        raise HTTPException(400, "Rating must be between 1 and 5")
    if len(text) > 2000:
        raise HTTPException(400, "Review text is too long (max 2000 characters)")

    saved = database.save_review(video_id, name[:100], title[:150], review.rating, text)
    return Review(**saved)


@app.get("/reviews/{video_id}", response_model=ReviewsResponse)
def list_reviews(video_id: str):
    """Returns all reviews for a route, plus the average rating and count."""
    rows = database.get_reviews(video_id)
    reviews = [Review(**r) for r in rows]
    count = len(reviews)
    average = round(sum(r.rating for r in reviews) / count, 1) if count else 0.0
    return ReviewsResponse(reviews=reviews, average_rating=average, count=count)


# ── Admin endpoints (basic — full panel comes later) ─────────────────────────

ADMIN_SECRET = os.getenv("ADMIN_SECRET", "getway2026")


@app.post("/admin/clear-cache")
def clear_cache(secret: str):
    """
    Deletes all cached itineraries so old TikTok links regenerate fresh.
    Protected by ADMIN_SECRET — set this in Railway's environment variables
    to something private; it falls back to a default if unset.
    """
    if secret != ADMIN_SECRET:
        raise HTTPException(403, "Invalid admin secret")
    count = database.clear_all_itineraries()
    return {"status": "ok", "cleared": count}


@app.post("/track/view/{video_id}")
def track_view(video_id: str):
    """
    Public, unauthenticated — bumps a route's view counter by 1. Called
    once when the AI route page loads. Silently no-ops (still returns ok)
    if the video_id doesn't exist, since a failed tracking ping should
    never surface an error to the visitor.
    """
    database.increment_view_count(video_id)
    return {"status": "ok"}


@app.post("/track/affiliate-click/{video_id}")
def track_affiliate_click(video_id: str):
    """
    Public, unauthenticated — bumps a route's affiliate-link-click counter
    by 1. Called when a visitor clicks a Booking.com/Expedia/Airbnb link.
    Note: this counts CLICKS, not confirmed bookings/commissions — actual
    commission revenue lives in the CJ Affiliate dashboard, not here.
    """
    database.increment_affiliate_click_count(video_id)
    return {"status": "ok"}


@app.get("/routes")
def list_public_routes():
    """
    Public, unauthenticated summary of every approved route — everything
    the homepage route grid needs (destination, duration, price category,
    tags, creator handle, stop count, hero photo) without exposing admin
    fields or requiring the admin secret. No pending/rejected routes here.

    If an admin has set featured_route_ids (via /admin/site-settings),
    only those routes are returned, in that exact order — this is how an
    admin manually curates the homepage instead of it always showing every
    approved route. Empty featured_route_ids (the default) means "show
    everything approved", unchanged from the original behavior.
    """
    all_approved = database.list_public_approved()
    featured_ids = database.get_site_settings().get("featured_route_ids", [])
    if not featured_ids:
        return all_approved
    by_id = {r["video_id"]: r for r in all_approved}
    return [by_id[vid] for vid in featured_ids if vid in by_id]


@app.get("/site-settings")
def get_public_site_settings():
    """Public, unauthenticated — homepage hero slides + featured routes."""
    return database.get_site_settings()


@app.put("/admin/site-settings")
def admin_update_site_settings(settings: SiteSettings, secret: str):
    """
    Sets the homepage's hero slide images and/or the admin-curated list of
    featured routes. Send an empty featured_route_ids list to go back to
    "show all approved routes automatically".
    """
    _check_admin_secret(secret)
    database.set_site_settings(settings.hero_slides, settings.featured_route_ids)
    return {"status": "ok"}


@app.get("/itinerary/{video_id}", response_model=ExtractResponse)
def get_itinerary(video_id: str):
    """
    Fetches a previously-generated itinerary by its stable video_id.
    Used to restore a shared route link (?route=<video_id>) on page load,
    since the frontend only has the ID at that point, not the original URL.
    """
    cached = database.get_itinerary(video_id)
    if not cached:
        raise HTTPException(404, "Itinerary not found for this route ID")
    return ExtractResponse(
        itinerary=cached,
        source="cache",
        video_id=video_id,
        cached=True,
    )


@app.get("/admin/itineraries")
def list_itineraries():
    """Lists all cached itineraries — useful for building the admin panel."""
    return database.list_itineraries()


def _check_admin_secret(secret: str) -> None:
    if secret != ADMIN_SECRET:
        raise HTTPException(403, "Invalid admin secret")


@app.get("/admin/pending")
def admin_list_pending(secret: str):
    """Full content of every route awaiting review — for the admin panel's Pending tab."""
    _check_admin_secret(secret)
    return database.list_by_status("pending")


@app.get("/admin/approved")
def admin_list_approved(secret: str):
    """Full content of every published route — for the admin panel's Published tab."""
    _check_admin_secret(secret)
    return database.list_by_status("approved")


@app.get("/admin/stats")
def admin_stats(secret: str):
    """Counts for the admin panel's Statistics tab."""
    _check_admin_secret(secret)
    return database.get_stats()


@app.post("/admin/approve/{video_id}")
def admin_approve(video_id: str, secret: str):
    """Marks a route as approved — makes it eligible for public display."""
    _check_admin_secret(secret)
    if not database.set_status(video_id, "approved"):
        raise HTTPException(404, "Route not found")
    return {"status": "ok", "video_id": video_id, "new_status": "approved"}


@app.post("/admin/reject/{video_id}")
def admin_reject(video_id: str, secret: str):
    """
    Marks a route as rejected (soft delete — kept in the DB so the
    Statistics tab can show a rejected count, and so a reject can be
    undone). Use DELETE /admin/route/{video_id} to permanently remove it.
    """
    _check_admin_secret(secret)
    if not database.set_status(video_id, "rejected"):
        raise HTTPException(404, "Route not found")
    return {"status": "ok", "video_id": video_id, "new_status": "rejected"}


@app.post("/admin/hide/{video_id}")
def admin_hide(video_id: str, secret: str):
    """Un-publishes an approved route back to pending (the panel's 'Скрий' button)."""
    _check_admin_secret(secret)
    if not database.set_status(video_id, "pending"):
        raise HTTPException(404, "Route not found")
    return {"status": "ok", "video_id": video_id, "new_status": "pending"}


@app.put("/admin/meta/{video_id}")
def admin_update_meta(video_id: str, meta: RouteMeta, secret: str):
    """
    Sets the homepage-grid curation fields (price category, filter tags,
    creator handle) an admin picks when curating a route — separate from
    the content editor since these don't come from the AI extraction.
    """
    _check_admin_secret(secret)
    if not database.set_route_meta(video_id, meta.price_category, meta.tags, meta.creator_handle):
        raise HTTPException(404, "Route not found")
    return {"status": "ok", "video_id": video_id}


@app.put("/admin/route/{video_id}")
def admin_update_route(video_id: str, itinerary: Itinerary, secret: str):
    """
    Overwrites a route's content (stops, hero/gallery photos, destination,
    duration) from the admin panel's inline editor. Status is untouched —
    use /admin/approve, /admin/reject, or /admin/hide for that.
    """
    _check_admin_secret(secret)
    if not database.update_itinerary_content(video_id, itinerary):
        raise HTTPException(404, "Route not found")
    return {"status": "ok", "video_id": video_id}


@app.delete("/admin/route/{video_id}")
def admin_delete_route(video_id: str, secret: str):
    """Permanently deletes a route (the panel's 'Изтрий' button)."""
    _check_admin_secret(secret)
    if not database.delete_itinerary_permanently(video_id):
        raise HTTPException(404, "Route not found")
    return {"status": "ok", "video_id": video_id, "deleted": True}


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0"}


# ── Static demo route photos ──────────────────────────────────────────────

_MALLORCA_PHOTO_QUERIES = {
    # Specific, vivid imagery terms rather than generic ones — this is a
    # fixed demo for one destination, so unlike the general AI pipeline
    # there's no need for queries that generalize across every city/island.
    # These target the shots Mallorca is actually known for: turquoise
    # coves, the cathedral's lake reflection, colorful old town streets.
    "hero": "Mallorca turquoise coast aerial",
    "gallery1": "Palma Cathedral reflection lake",
    "gallery2": "Mallorca turquoise cove beach",
    "gallery3": "Mallorca old town colorful street",
    "gallery4": "Mallorca coastal cliffs sunset",
    "palma_cathedral": "Palma Cathedral La Seu Mallorca",
    "restaurant_illeta": "Mallorca seaside restaurant turquoise water",
    "valldemossa": "Valldemossa Mallorca stone village",
    "beach_calo_del_moro": "Cala del Moro Mallorca turquoise",
    "beach_salmunia": "S'Almunia Mallorca beach turquoise",
    "beach_cala_llombards": "Cala Llombards Mallorca turquoise",
    "beach_platja_santanyi": "Platja de Santanyi Mallorca turquoise",
    "deia": "Deia Mallorca village mountains",
    "hotel": "luxury boutique hotel pool Mallorca",
    "hotel2": "beachfront resort infinity pool Mallorca",
}


_mallorca_photos_cache: dict | None = None


@app.get("/demo/mallorca-photos")
def get_mallorca_demo_photos():
    """
    Real photos for the static homepage Mallorca demo route, fetched from
    Unsplash server-side (keeps the API key off the client — this is a
    plain GET the frontend can call directly). Replaces the old Lorem
    Picsum placeholder images: Picsum's "seed" is just a random-photo seed,
    not a content filter, so "mallorca-g1" never actually returned a photo
    of Mallorca.

    Cached in memory after the first call: this content is fixed (a demo
    route, not a live search), so there's no reason to spend Unsplash's
    50-requests/hour free-tier quota again for every single homepage visit
    — that quota is needed for real AI-generated routes.
    """
    global _mallorca_photos_cache
    if _mallorca_photos_cache is not None:
        return _mallorca_photos_cache

    result = {}
    for key, query in _MALLORCA_PHOTO_QUERIES.items():
        candidates = sorted(
            _unsplash_candidates(query, per_page=10),
            key=lambda r: r.get("likes", 0), reverse=True,
        )
        if candidates:
            best = candidates[0]
            _trigger_unsplash_download(best)
            result[key] = {
                "url": best.get("urls", {}).get("regular", ""),
                "attribution": _attribution_from_candidate(best),
            }
        else:
            result[key] = {"url": "", "attribution": None}

    # Only cache a mostly-successful result. If Unsplash was rate-limited
    # (or the key is unset) right after a restart, most/all entries come
    # back "" — caching that would permanently serve blank photos until
    # the next deploy. Leaving it uncached lets the next request retry.
    found = sum(1 for v in result.values() if v.get("url"))
    if found >= len(_MALLORCA_PHOTO_QUERIES) // 2:
        _mallorca_photos_cache = result
    else:
        print(f"[Demo] Only found {found}/{len(_MALLORCA_PHOTO_QUERIES)} Mallorca photos — not caching, will retry next request")

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
