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

import json
import tempfile
import threading
from datetime import datetime, timezone

import requests

# Load .env FIRST — before any module that reads ANTHROPIC_API_KEY
from dotenv import load_dotenv
load_dotenv()

import os

import uuid

from fastapi import FastAPI, HTTPException, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from models import (
    ExtractRequest, ExtractResponse, Itinerary, DayPlan, Comment, ReviewCreate, Review,
    ReviewsResponse, RouteMeta, SiteSettings, TripCandidate, TripCandidatesRequest,
    TripCandidatesResponse, TripHotelRequest, TripHotelRecommendation, TripHotelResponse,
    TripBuildRequest, TripSaveRequest, TripSaveResponse, TripEditStateResponse,
)
import database
from extractor import (
    extract_video_id, fetch_metadata, download_video, extract_frames,
    is_slideshow, fetch_slideshow_post, download_slideshow_images,
    fetch_top_comments, is_instagram_url, fetch_instagram_post,
    download_instagram_video, resolve_canonical_url,
)
from troll_filter import check_is_travel
from ai_analyzer import analyse_frames, generate_fun_fact
from quality_check import ai_quality_check
from places import (
    enrich_itinerary_with_photos, _unsplash_candidates, _attribution_from_candidate,
    _trigger_unsplash_download, _get_place_photo_and_location, search_attractions_broad,
    search_hotels_near, _get_destination_gallery_unsplash,
)
from trip_builder import (
    is_open as trip_place_is_open, fits_budget, curate_candidates,
    cluster_center, pick_hotel, hotel_to_recommendation_dict,
    assemble_days, recommend_car_rental,
)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="GetWay Backend", version="0.2.0")

# Admin-uploaded images (hero/gallery/stop photos) are stored directly on
# Railway's persistent Volume — same one the database already lives on —
# instead of a third-party host like imgbb. This survives redeploys (same
# reason the DB was moved there) and means GetWay no longer depends on a
# free external image host that can silently swap a missing photo for its
# own "image not found" placeholder. See DATA_DIR in database.py for why
# this path is safe to write to.
IMAGES_DIR = database.DATA_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Public base URL this backend is reachable at — used to build the photo
# URLs returned to the admin panel (e.g. "https://xxx.up.railway.app/uploads/abc.jpg").
# Set BACKEND_BASE_URL in Railway → Variables if this ever changes.
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "https://getway-production.up.railway.app").rstrip("/")

# Serves everything in IMAGES_DIR at /uploads/<filename> — this is what
# turns an uploaded file into a working photo URL for the site.
app.mount("/uploads", StaticFiles(directory=str(IMAGES_DIR)), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://gogetway.com",
        "https://www.gogetway.com",
        "https://getway-theta.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "*",            # tighten this before production launch
    ],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


def _backfill_fun_facts_background():
    """
    Fills `fun_fact` (see models.Itinerary.fun_fact) for every already-
    approved route that doesn't have one yet, so the homepage fact chip
    (index.html) starts showing up on existing routes without needing to
    re-run the full video analysis. Runs once at every startup, in a
    background thread, so it never delays the app becoming ready to serve
    requests. Cheap (one Haiku call per missing route — a few cents total
    for a small catalog) and safe to run on every restart: already-filled
    routes are skipped by list_approved_missing_fun_fact(), so once the
    catalog is fully backfilled this becomes a fast no-op on later starts.
    New routes never need this — they get their fun_fact for free as part
    of the normal Sonnet analysis pass (analyse_frames in ai_analyzer.py).
    """
    try:
        targets = database.list_approved_missing_fun_fact()
        if not targets:
            return
        print(f"[FunFact] Backfilling {len(targets)} route(s) missing a fun_fact...")
        filled = 0
        for row in targets:
            fact, _cost = generate_fun_fact(row["destination"])
            if fact:
                database.set_fun_fact(row["video_id"], fact)
                filled += 1
        print(f"[FunFact] Backfill done — {filled}/{len(targets)} filled")
    except Exception as e:
        print(f"[FunFact] Background backfill crashed (non-fatal): {e}")


@app.on_event("startup")
def startup():
    database.init_db()
    print("[Startup] GetWay backend ready")
    threading.Thread(target=_backfill_fun_facts_background, daemon=True).start()


# ── Main extraction endpoint ──────────────────────────────────────────────────

@app.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest):
    url = req.url.strip()
    if not url.startswith("http"):
        raise HTTPException(400, "Invalid URL — must start with http or https")

    # Resolve TikTok short share links (vm.tiktok.com/..., tiktok.com/t/...)
    # to their canonical form BEFORE computing video_id — otherwise the
    # same video shared/copied twice can get two different short links
    # (TikTok embeds a per-share tracking code), missing the cache and
    # triggering a second paid generation for content already on file.
    url = resolve_canonical_url(url)

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
            meta = {
                "title": slideshow_data["title"],
                "description": slideshow_data["description"],
                "uploader": slideshow_data.get("uploader", ""),
            }
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

    # ── Layer 9: AI Quality Check — Claude Haiku (~$0.001, non-fatal) ─────────
    # Runs automatically on every freshly generated route so a score is
    # already sitting on the card in the admin panel's Pending tab, with no
    # need to click "🤖 AI Check" manually. This only flags issues — it
    # never approves/rejects/publishes anything; that's always Gerry's call.
    try:
        qc_result = ai_quality_check(itinerary)
        database.save_quality_check(video_id, qc_result)
        print(f"[QualityCheck] {video_id} → score={qc_result['score']} status={qc_result['status']}")
    except Exception as e:
        print(f"[QualityCheck] Failed (non-fatal): {e}")

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
    itinerary.source_url = url

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


# ── Build Your Own Trip (Phase A: candidate search) ───────────────────────────

_TRIP_BUDGET_TIERS = {"cheap", "mid", "luxury"}
_TRIP_CACHE_TTL_DAYS = 30


@app.post("/trip/candidates", response_model=TripCandidatesResponse)
def get_trip_candidates(req: TripCandidatesRequest):
    """
    "Build Your Own Trip" secondary feature — Phase A (backend only, no
    frontend yet). Returns a curated, budget-appropriate list of candidate
    attractions for a destination: a broad Places search
    (places.search_attractions_broad), filtered for open/budget-fit
    (trip_builder.is_open / fits_budget), then run through an AI curation
    pass (trip_builder.curate_candidates) that drops junk/duplicates and
    rewrites descriptions in site tone. Cached by (city, budget) — see
    database.get_trip_candidates_cache/save_trip_candidates_cache.
    """
    city = req.destination.strip()
    budget = req.budget.strip().lower()
    if not city:
        raise HTTPException(400, "Missing destination")
    if budget not in _TRIP_BUDGET_TIERS:
        raise HTTPException(400, f"budget must be one of: {', '.join(sorted(_TRIP_BUDGET_TIERS))}")

    cached = database.get_trip_candidates_cache(city, budget)
    if cached is not None:
        print(f"[TripBuilder] Cache HIT for {city} / {budget} ({len(cached)} candidates)")
        return TripCandidatesResponse(
            destination=city, budget=budget,
            candidates=[TripCandidate(**c) for c in cached],
            cached=True,
        )

    print(f"[TripBuilder] Cache MISS for {city} / {budget} — searching Places")
    raw_places = search_attractions_broad(city)
    fitted = [p for p in raw_places if trip_place_is_open(p) and fits_budget(p, budget)]
    curated, cost_usd = curate_candidates(city, fitted)
    print(f"[TripBuilder] {city}/{budget}: {len(raw_places)} raw -> {len(fitted)} fit budget -> "
          f"{len(curated)} after AI curation (${cost_usd:.4f})")

    database.save_trip_candidates_cache(city, budget, curated, ttl_days=_TRIP_CACHE_TTL_DAYS)
    return TripCandidatesResponse(
        destination=city, budget=budget,
        candidates=[TripCandidate(**c) for c in curated],
        cached=False,
    )


@app.post("/trip/hotel", response_model=TripHotelResponse)
def get_trip_hotel(req: TripHotelRequest):
    """
    "Build Your Own Trip" Phase B — recommends a single hotel matching the
    chosen budget tier, geographically anchored to whichever attractions
    the traveler selected from /trip/candidates. LOCKED LOGIC: the hotel is
    always open and 4.0+ rated regardless of budget (see
    trip_builder.pick_hotel) — "cheap" only changes the room-rate tier, not
    location quality. Cached by (city, budget, rough selection centroid) —
    see database.get_trip_hotel_cache/save_trip_hotel_cache.
    """
    city = req.destination.strip()
    budget = req.budget.strip().lower()
    if not city:
        raise HTTPException(400, "Missing destination")
    if budget not in _TRIP_BUDGET_TIERS:
        raise HTTPException(400, f"budget must be one of: {', '.join(sorted(_TRIP_BUDGET_TIERS))}")
    if not req.selected_attractions:
        raise HTTPException(400, "selected_attractions is required to anchor the hotel search")

    center = cluster_center([(a.lat, a.lng) for a in req.selected_attractions])
    if center is None:
        raise HTTPException(400, "selected_attractions had no usable coordinates")
    lat, lng = center

    cached = database.get_trip_hotel_cache(city, budget, lat, lng)
    if cached is not None:
        print(f"[TripBuilder] Hotel cache HIT for {city} / {budget} near ({lat:.2f}, {lng:.2f})")
        return TripHotelResponse(
            destination=city, budget=budget,
            hotel=TripHotelRecommendation(**cached) if cached else None,
            cached=True,
        )

    print(f"[TripBuilder] Hotel cache MISS for {city} / {budget} near ({lat:.2f}, {lng:.2f}) — searching Places")
    raw_hotels = search_hotels_near(lat, lng, city)
    chosen = pick_hotel(raw_hotels, budget)
    hotel_dict = hotel_to_recommendation_dict(chosen, city) if chosen else None
    print(f"[TripBuilder] {city}/{budget}: {len(raw_hotels)} raw hotels -> "
          f"{'picked ' + hotel_dict['name'] if hotel_dict else 'none qualified (4.0+ rating)'}")

    database.save_trip_hotel_cache(city, budget, lat, lng, hotel_dict, ttl_days=_TRIP_CACHE_TTL_DAYS)
    return TripHotelResponse(
        destination=city, budget=budget,
        hotel=TripHotelRecommendation(**hotel_dict) if hotel_dict else None,
        cached=False,
    )


_BUDGET_PRICE_CATEGORY = {"cheap": "€", "mid": "€€", "luxury": "€€€"}


@app.post("/trip/build", response_model=ExtractResponse)
def build_trip(req: TripBuildRequest):
    """
    "Build Your Own Trip" Phase C — assembles a full Itinerary from the
    traveler's selected attractions, in the SAME Stop/DayPlan/Itinerary
    shape the video-generated routes already use (trip_builder.assemble_days),
    including a Phase B hotel recommendation and a car rental recommendation
    from real-world destination knowledge (trip_builder.recommend_car_rental
    — NOT a distance heuristic). Nothing is saved to the database here —
    this only builds and returns the object for a frontend preview; saving
    is a later phase.

    source="custom_builder" in the response — deliberately distinct from
    "ai_generated"/"cache" so nothing downstream that branches on source
    confuses a self-built trip with a video-extracted route.
    """
    city = req.destination.strip()
    budget = req.budget.strip().lower()
    if not city:
        raise HTTPException(400, "Missing destination")
    if budget not in _TRIP_BUDGET_TIERS:
        raise HTTPException(400, f"budget must be one of: {', '.join(sorted(_TRIP_BUDGET_TIERS))}")
    if req.days < 1:
        raise HTTPException(400, "days must be at least 1")
    if not req.selected_attractions:
        raise HTTPException(400, "selected_attractions must not be empty")

    selected_dicts = [a.model_dump() for a in req.selected_attractions]

    # Hotel: same cluster-anchored search + cache as /trip/hotel, called
    # directly here (not a second HTTP round-trip) so a build always
    # reflects exactly the same hotel logic a separate preview call would
    # have shown for this same selection.
    center = cluster_center([(a.lat, a.lng) for a in req.selected_attractions])
    hotel_dict = None
    if center is not None:
        lat, lng = center
        cached_hotel = database.get_trip_hotel_cache(city, budget, lat, lng)
        if cached_hotel is not None:
            hotel_dict = cached_hotel or None
        else:
            raw_hotels = search_hotels_near(lat, lng, city)
            chosen = pick_hotel(raw_hotels, budget)
            hotel_dict = hotel_to_recommendation_dict(chosen, city) if chosen else None
            database.save_trip_hotel_cache(city, budget, lat, lng, hotel_dict, ttl_days=_TRIP_CACHE_TTL_DAYS)

    days = assemble_days(selected_dicts, req.days, hotel_dict)

    car_recommended, car_note, car_cost_usd = recommend_car_rental(city)
    print(f"[TripBuilder] Car rental for {city}: recommended={car_recommended} (${car_cost_usd:.4f})")

    itinerary = Itinerary(
        destination=city,
        duration=f"{req.days} day{'s' if req.days != 1 else ''}",
        days=[DayPlan(**d) for d in days],
        price_category=_BUDGET_PRICE_CATEGORY.get(budget, ""),
        car_rental_recommended=car_recommended,
        car_rental_note=car_note,
    )

    # Hero/gallery: reuse the existing destination Unsplash logic exactly
    # as-is (places._get_destination_gallery_unsplash — the same function
    # enrich_itinerary_with_photos() itself calls for this). Deliberately
    # NOT calling enrich_itinerary_with_photos() wholesale here: that
    # function also re-fetches a fresh Places photo for every STOP, which
    # would waste API calls re-confirming photos the Phase A/B searches
    # already gave us and risks swapping in a different (not necessarily
    # better) match than the one the traveler saw and picked.
    try:
        gallery = _get_destination_gallery_unsplash(city, count=1)
        itinerary.gallery_photo_urls = [g["url"] for g in gallery]
        itinerary.gallery_attributions = [g["attribution"] for g in gallery]
        best = max(gallery, key=lambda g: g["likes"]) if gallery else None
        itinerary.hero_photo_url = best["url"] if best else ""
        itinerary.hero_attribution = best["attribution"] if best else None
    except Exception as e:
        print(f"[TripBuilder] Hero/gallery photo fetch failed (non-fatal): {e}")

    return ExtractResponse(
        itinerary=itinerary,
        source="custom_builder",
        video_id="",
        cached=False,
    )


@app.post("/trip/save", response_model=TripSaveResponse)
def save_trip(req: TripSaveRequest):
    """
    Persists a built custom trip (the output of /trip/build) so it's
    reachable via a shareable slug (GET /trip/{slug}). Pass `slug` to
    update an existing saved trip in place instead of creating a new one —
    the edit flow: GET /trip/{slug}/edit-state to reopen the builder
    pre-filled, POST /trip/build again with the traveler's changes, then
    POST here with that SAME slug.

    No auth/ownership check — matches the rest of the site, which has no
    user accounts at all; anyone with the slug can view or re-save it.
    """
    budget = req.budget.strip().lower()
    if budget not in _TRIP_BUDGET_TIERS:
        raise HTTPException(400, f"budget must be one of: {', '.join(sorted(_TRIP_BUDGET_TIERS))}")
    if not req.selected_attractions:
        raise HTTPException(400, "selected_attractions must not be empty")

    slug = database.save_custom_trip(
        destination=req.destination.strip(),
        days=req.days,
        people=req.people,
        budget=budget,
        selected_attractions=[a.model_dump() for a in req.selected_attractions],
        itinerary=req.itinerary,
        slug=req.slug,
    )
    return TripSaveResponse(slug=slug)


@app.get("/trip/{slug}", response_model=ExtractResponse)
def get_saved_trip(slug: str):
    """
    Returns a saved custom trip in the SAME ExtractResponse shape as
    video-generated routes (see GET /itinerary/{video_id}) — so the
    frontend's route-rendering code doesn't need a separate path for
    builder-made trips. video_id is set to the slug so any generic code
    that just needs SOME route identifier keeps working unmodified.
    """
    saved = database.get_custom_trip(slug)
    if not saved:
        raise HTTPException(404, "Trip not found for this slug")
    return ExtractResponse(
        itinerary=Itinerary(**saved["itinerary"]),
        source="custom_builder",
        video_id=slug,
        cached=True,
    )


@app.get("/trip/{slug}/edit-state", response_model=TripEditStateResponse)
def get_trip_edit_state(slug: str):
    """
    Returns the ORIGINAL builder inputs for a saved trip — destination,
    days, people, budget, and the exact selected_attractions — everything
    needed to reopen the builder pre-filled. The saved GET /trip/{slug}
    page itself stays a locked, non-editable snapshot; this is what a
    future "Edit this trip" button calls before showing the builder form
    again (see POST /trip/save's `slug` param for the resulting resave).
    """
    saved = database.get_custom_trip(slug)
    if not saved:
        raise HTTPException(404, "Trip not found for this slug")
    return TripEditStateResponse(
        slug=slug,
        destination=saved["destination"],
        days=saved["days"],
        people=saved["people"],
        budget=saved["budget"],
        selected_attractions=[TripCandidate(**a) for a in saved["selected_attractions"]],
    )


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


_ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15MB — generous for a phone photo


@app.post("/admin/upload-image")
async def admin_upload_image(secret: str, file: UploadFile = File(...)):
    """
    Saves an uploaded image file (hero/gallery/stop photo) directly onto
    Railway's persistent Volume and returns a direct URL served by this
    same backend — lets the admin panel offer a real drag-and-drop/
    file-picker upload without depending on a third-party host.

    Previously this uploaded to imgbb.com. That's been retired: imgbb is a
    free host with no reliability guarantee, and — worse — when an imgbb
    image goes missing it doesn't return a 404, it silently serves back
    its own "image not found" placeholder graphic with a 200 OK, so the
    site's own broken-image fallback never even triggers. Storing files
    ourselves removes that failure mode entirely: the photo is only ever
    gone if this server's Volume is gone, same as the database already
    depends on.
    """
    _check_admin_secret(secret)

    content_type = (file.content_type or "").lower()
    extension = _ALLOWED_IMAGE_TYPES.get(content_type)
    if not extension:
        raise HTTPException(
            415,
            f"Unsupported file type '{content_type}'. "
            f"Allowed: {', '.join(sorted(_ALLOWED_IMAGE_TYPES))}.",
        )

    contents = await file.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"Image is larger than the {_MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.",
        )
    if not contents:
        raise HTTPException(400, "Uploaded file is empty.")

    # Random filename — never trust the original filename (path traversal,
    # collisions, weird characters) — same principle as the slugify() rule
    # already used elsewhere for route names.
    filename = f"{uuid.uuid4().hex}{extension}"
    file_path = IMAGES_DIR / filename

    try:
        file_path.write_bytes(contents)
    except OSError as e:
        raise HTTPException(500, f"Could not save image: {e}")

    return {"url": f"{BACKEND_BASE_URL}/uploads/{filename}"}


@app.post("/admin/quality-check/{video_id}")
def admin_quality_check(video_id: str, secret: str):
    """
    The admin panel's ONLY "🤖 AI Check" button — does everything in one
    click, one cost:

      1. Auto-fixes what can be fixed: for stops missing coordinates or a
         real photo, re-runs Google Places and backfills whatever it can
         confidently resolve. Only calls Places for BROKEN stops — a route
         with no issues costs zero Places calls, and each fixable stop
         costs exactly one call (not two, from folding the old separate
         "confirm" + "fetch" steps into a single _get_place_photo_and_location
         call).
      2. Runs the quality check (Claude Haiku, ~$0.001) against the
         now-updated data — generic names, coordinate/country plausibility,
         plus the free deterministic checks.
      3. Saves both the fixed stop data (days_json) and the check result
         (qc_json).

    This used to be two separate buttons/endpoints ("Verify & fix
    locations" + "AI Check") — merged into one, since running them
    separately just meant paying for two API round-trips to get one
    useful answer. A stop Places still can't confidently match (usually a
    genuinely generic name like "Trastevere neighbourhood" rather than a
    specific place) can't be auto-fixed by any amount of re-running this —
    it needs a manual rename via Edit, then one more click of this same
    button.

    This ONLY flags/fixes location data — it never changes `status`.
    Approving or rejecting the route is always a separate, manual action
    via /admin/approve or /admin/reject.

    Returns the same shape as ai_quality_check() plus "fixed_count".
    """
    _check_admin_secret(secret)
    itinerary = database.get_itinerary(video_id)
    if itinerary is None:
        raise HTTPException(404, "Route not found")

    city = itinerary.destination.split(",")[0].strip()
    fixed_count = 0
    changed = False

    for day in itinerary.days:
        for stop in day.stops:
            needs_coords = stop.lat is None or stop.lng is None
            needs_photo = not stop.photo_url or "picsum.photos" in stop.photo_url.lower()
            if not (needs_coords or needs_photo):
                continue  # already fine — zero Places calls spent on it

            query = stop.name if city.lower() in stop.name.lower() else f"{stop.name}, {city}"
            photo_url, location = _get_place_photo_and_location(query)

            stop_fixed = False
            if location and needs_coords:
                stop.lat, stop.lng = location
                stop_fixed = True
            if photo_url and needs_photo:
                stop.photo_url = photo_url
                stop_fixed = True
            if stop_fixed:
                fixed_count += 1
                changed = True

    if changed:
        database.save_days(video_id, itinerary.days)

    result = ai_quality_check(itinerary)
    result["fixed_count"] = fixed_count
    database.save_quality_check(video_id, result)
    return result


@app.post("/admin/backfill-fun-facts")
def admin_backfill_fun_facts(secret: str, force: bool = False):
    """
    Manually re-triggers the same fun_fact backfill that already runs
    automatically at every startup (see _backfill_fun_facts_background) —
    useful to get an immediate result/count right after approving a batch
    of new routes, instead of waiting for the next deploy/restart. Safe to
    call any time; by default routes that already have a fun_fact are
    skipped.

    force=true instead re-generates EVERY approved route's fun_fact, even
    ones that already have one — use this once after a prompt change (e.g.
    asking for more surprising/quirky facts instead of generic ones) to
    upgrade facts that were already written under the old prompt, not just
    fill in blanks. Costs roughly $0.0005 x number of approved routes.

    Returns how many were checked, how many got filled, and the real
    Anthropic cost of this run.
    """
    _check_admin_secret(secret)
    targets = database.list_approved_for_fun_fact_refresh(force=force)
    filled = 0
    total_cost = 0.0
    for row in targets:
        fact, cost = generate_fun_fact(row["destination"])
        total_cost += cost
        if fact:
            database.set_fun_fact(row["video_id"], fact)
            filled += 1
    return {"status": "ok", "checked": len(targets), "filled": filled, "cost_usd": round(total_cost, 6)}


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


@app.get("/admin/export")
def admin_export_routes(secret: str):
    """
    Dumps every approved route as a downloadable JSON file.

    Used by the "Export all routes" button in admin.html, and by the
    nightly backup GitHub Action (.github/workflows/backup.yml) which
    hits this endpoint and commits the result to backups/ in the repo —
    a second copy of the data independent of the Railway Volume.
    """
    _check_admin_secret(secret)
    routes = database.list_by_status("approved")
    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "count": len(routes),
        "routes": routes,
    }
    body = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    filename = f"routes-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.json"
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/admin/approve/{video_id}")
def admin_approve(video_id: str, secret: str):
    """Marks a route as approved — makes it eligible for public display."""
    _check_admin_secret(secret)
    if not database.set_status(video_id, "approved"):
        raise HTTPException(404, "Route not found")
    # Approval is a strong signal the stops' current photos/addresses are
    # good — save them for reuse if this same city gets another route
    # generated later (e.g. a second Rome video also featuring the Colosseum).
    itinerary = database.get_itinerary(video_id)
    if itinerary:
        database.cache_stops_from_itinerary(itinerary.destination, itinerary)
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
    # A manual save is an even stronger signal than approval that these
    # specific photos/addresses are the right ones — cache them too.
    database.cache_stops_from_itinerary(itinerary.destination, itinerary)
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


# ── Rankings section photos (homepage "Most visited destinations") ────────

_RANKING_PHOTO_QUERIES = {
    # Keys match the `key` field on each entry in index.html's
    # GLOBAL_TOURISM_RANKINGS / GLOBAL_TOURISM_VOLUME_HIGHLIGHT — this is
    # fixed editorial content (the Rankings list), refreshed by hand once a
    # year alongside the ranking data itself, not a live per-visitor search.
    "paris-france": "Paris Eiffel Tower cityscape",
    "madrid-spain": "Madrid Gran Via street",
    "tokyo-japan": "Tokyo Shibuya crossing skyline",
    "rome-italy": "Rome Colosseum aerial",
    "milan-italy": "Milan Duomo cathedral",
    "bangkok-thailand": "Bangkok Wat Arun temple skyline",
}

_ranking_photos_cache: dict | None = None


@app.get("/demo/rankings-photos")
def get_rankings_photos():
    """
    Real photos for the homepage Rankings section's fixed city list,
    fetched from Unsplash server-side — same pattern as
    /demo/mallorca-photos (keeps the API key off the client, cached in
    memory after the first mostly-successful call since this is fixed
    editorial content, not a live search).
    """
    global _ranking_photos_cache
    if _ranking_photos_cache is not None:
        return _ranking_photos_cache

    result = {}
    for key, query in _RANKING_PHOTO_QUERIES.items():
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

    found = sum(1 for v in result.values() if v.get("url"))
    if found >= len(_RANKING_PHOTO_QUERIES) // 2:
        _ranking_photos_cache = result
    else:
        print(f"[Demo] Only found {found}/{len(_RANKING_PHOTO_QUERIES)} ranking photos — not caching, will retry next request")

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
