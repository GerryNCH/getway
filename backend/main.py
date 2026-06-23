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

from models import ExtractRequest, ExtractResponse, Itinerary
import database
from extractor import extract_video_id, fetch_metadata, download_video, extract_frames
from troll_filter import check_is_travel
from ai_analyzer import analyse_frames

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
    allow_methods=["GET", "POST"],
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
    try:
        meta = fetch_metadata(url)
        print(f"[Meta] Title: {meta['title'][:60]}")
    except RuntimeError as e:
        raise HTTPException(422, f"Could not fetch video info: {e}")

    # ── Layer 4: troll filter — Claude Haiku (~$0.0003) ───────────────────────
    is_travel, reason = check_is_travel(
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

        # Download video
        try:
            video_path = download_video(url, tmp)
            print(f"[Download] {video_path}")
        except RuntimeError as e:
            raise HTTPException(422, f"Video download failed: {e}")

        # Extract frames
        try:
            frames = extract_frames(video_path, tmp, req.max_frames)
            print(f"[Frames] Extracted {len(frames)} frames")
        except RuntimeError as e:
            raise HTTPException(500, f"Frame extraction failed: {e}")

        # Claude multimodal analysis
        try:
            itinerary: Itinerary = analyse_frames(frames)
            print(f"[AI] Destination: {itinerary.destination} — "
                  f"{sum(len(d.stops) for d in itinerary.days)} stops across "
                  f"{len(itinerary.days)} days")
        except Exception as e:
            raise HTTPException(500, f"AI analysis failed: {e}")

    # ── Layer 8: save to database ─────────────────────────────────────────────
    database.save_itinerary(video_id, url, itinerary)

    return ExtractResponse(
        itinerary=itinerary,
        source="ai_generated",
        video_id=video_id,
        cached=False,
    )


# ── Admin endpoints (basic — full panel comes later) ─────────────────────────

@app.get("/admin/itineraries")
def list_itineraries():
    """Lists all cached itineraries — useful for building the admin panel."""
    return database.list_itineraries()


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
