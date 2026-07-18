"""
extractor.py — Video download and frame extraction.
Zero system dependencies: uses yt-dlp (pip) + opencv-python (pip) only.
No ffmpeg, no ffprobe, no brew installs required.
"""

import json
import os
import re
import subprocess
from pathlib import Path

import cv2       # pip install opencv-python
import requests  # pip install requests


# ── Video ID normalisation ────────────────────────────────────────────────────

def extract_video_id(url: str) -> str:
    """Stable, platform-agnostic video ID used as the database key."""
    m = re.search(r"/(?:video|photo)/(\d+)", url)
    if m:
        return f"tt_{m.group(1)}"
    m = re.search(r"instagram\.com/(?:[\w.]+/)?(?:reel|reels|p)/([A-Za-z0-9_-]+)", url)
    if m:
        return f"ig_{m.group(1)}"
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if m:
        return f"yt_{m.group(1)}"
    import hashlib
    return "u_" + hashlib.md5(url.encode()).hexdigest()[:16]


# ── Slideshow detection ───────────────────────────────────────────────────────

def is_slideshow(url: str) -> bool:
    """
    TikTok slideshow posts use /photo/ URLs (a series of images) instead of
    /video/ URLs. These need a different download path since there's no
    video stream to pull frames from.
    """
    return "/photo/" in url


def is_instagram_url(url: str) -> bool:
    """True for an Instagram Reel or post link (instagram.com/reel/, /reels/, or /p/)."""
    url = url.lower()
    return "instagram.com" in url and any(seg in url for seg in ("/reel/", "/reels/", "/p/"))


# ── Metadata fetch (no download) ─────────────────────────────────────────────

def fetch_metadata(url: str) -> dict:
    """
    Calls yt-dlp --dump-json to get title + description without downloading.
    Fast (1-3 seconds). yt-dlp is a pure Python pip package.
    """
    result = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-download",
         "--no-playlist", "--quiet", url],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not fetch video metadata: {result.stderr.strip()}"
        )
    data = json.loads(result.stdout)
    return {
        "title":       data.get("title", ""),
        "description": data.get("description", ""),
        "uploader":    data.get("uploader", ""),
        "duration":    data.get("duration", 0),
        "webpage_url": data.get("webpage_url", url),
    }


# ── Video download ────────────────────────────────────────────────────────────

def download_video(url: str, output_dir: str) -> str:
    """
    Downloads the video via yt-dlp at ≤720p as mp4.
    yt-dlp is a pure Python package — no system tools needed for this step.
    """
    output_template = os.path.join(output_dir, "video.%(ext)s")
    result = subprocess.run(
        [
            "yt-dlp",
            "--format",
            # Request single-file formats only — no audio+video merging needed.
            # mp4 with video+audio in one file, no ffmpeg required.
            "best[ext=mp4][height<=720]/best[ext=mp4]/best[height<=720]/best",
            "--output", output_template,
            "--no-playlist",
            "--no-part",
            "--quiet",
            url,
        ],
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed: {result.stderr.strip()}")

    matches = list(Path(output_dir).glob("video.*"))
    if not matches:
        raise RuntimeError("Download completed but no video file was found.")
    return str(matches[0])


# ── Slideshow post fetch (metadata + images via Apify) ───────────────────────
#
# yt-dlp has NO support for TikTok /photo/ (slideshow) posts — confirmed by
# reading yt-dlp's own tiktok.py source (its URL regex only matches /video/)
# and by checking yt-dlp's GitHub issue tracker: "[TikTok] Support for
# Photos" (#9990) and "Add photo album download support" (#8360) were both
# closed by the maintainers as "wontfix". This isn't a version problem —
# no yt-dlp update will fix it.
#
# Instead, slideshow posts go through the Apify "TikTok Scraper" actor
# (clockworks/tiktok-scraper) — a well-established, actively maintained
# actor with a built-in "download slideshow images" option. One request
# returns both the post's caption (for the troll filter) and the slide
# image URLs, so we only call it once per slideshow.
#
# Requires APIFY_API_TOKEN set in the environment (Railway → Variables).

APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN")
APIFY_TIKTOK_ACTOR = "clockworks~tiktok-scraper"


def fetch_slideshow_post(url: str) -> dict:
    """
    Single Apify call for a TikTok slideshow post. Returns everything the
    pipeline needs: title/description (for the troll filter), uploader
    (creator handle), and the ordered list of slide image URLs (for the
    AI analysis step).

    Field names below were confirmed against real run output (not
    guessed): each dataset item has `text` (the caption),
    `slideshowImageLinks` (list of {"tiktokLink", "downloadLink"} objects),
    and — with `scrapeAdditionalAuthorMeta: true` in the request —
    `authorMeta.name` (a flat key literally containing a dot, not a nested
    object). We use `downloadLink` for images — it's hosted on Apify's own
    storage and doesn't expire, unlike the raw signed `tiktokLink` CDN URL.
    """
    if not APIFY_API_TOKEN:
        raise RuntimeError(
            "APIFY_API_TOKEN is not set — add it in Railway → Variables."
        )

    endpoint = f"https://api.apify.com/v2/acts/{APIFY_TIKTOK_ACTOR}/run-sync-get-dataset-items"
    payload = {
        "postURLs": [url],
        "shouldDownloadSlideshowImages": True,
        "shouldDownloadVideos": False,
        "shouldDownloadCovers": False,
        "shouldDownloadSubtitles": False,
        "shouldDownloadAvatars": False,
        "shouldDownloadMusicCovers": False,
        "scrapeAdditionalAuthorMeta": True,  # needed for authorMeta.name (creator handle) —
                                               # confirmed field name from a real test run
    }

    try:
        resp = requests.post(
            endpoint, params={"token": APIFY_API_TOKEN}, json=payload, timeout=90,
        )
        resp.raise_for_status()
        items = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Apify request failed: {e}")

    if not items:
        raise RuntimeError(
            "Apify returned no data for this post — make sure the URL is public."
        )

    post = items[0]
    slideshow_links = post.get("slideshowImageLinks") or []
    image_urls = [
        link.get("downloadLink") or link.get("tiktokLink")
        for link in slideshow_links
        if link.get("downloadLink") or link.get("tiktokLink")
    ]

    if not image_urls:
        raise RuntimeError(
            "No slideshow images returned. Either this post isn't a slideshow, "
            "or Apify's output field name has changed — check the Actor's API "
            "tab on apify.com for the current field name."
        )

    caption = post.get("text", "")
    return {
        "title": caption,
        "description": caption,
        "uploader": post.get("authorMeta.name", ""),
        "webpage_url": post.get("webVideoUrl", url),
        "image_urls": image_urls,
    }


def download_slideshow_images(image_urls: list[str], output_dir: str) -> list[str]:
    """
    Downloads slide images whose URLs were already fetched by
    fetch_slideshow_post(). Kept separate so the Apify call only happens
    once even though we need the images after the troll filter check.

    Apify's key-value-store download links are normally public, but if a
    request comes back 401/403 we retry once with the API token attached
    (?token=...) in case the store needs auth.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; GetWayBot/1.0)"}
    image_paths: list[str] = []

    for i, img_url in enumerate(image_urls):
        try:
            resp = requests.get(img_url, headers=headers, timeout=20)
            if resp.status_code in (401, 403) and APIFY_API_TOKEN:
                resp = requests.get(
                    img_url, headers=headers, timeout=20,
                    params={"token": APIFY_API_TOKEN},
                )
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[Slideshow] Skipping image {i} — download failed: {e}")
            continue

        out_path = os.path.join(output_dir, f"slide_{i:02d}.jpg")
        with open(out_path, "wb") as f:
            f.write(resp.content)
        image_paths.append(out_path)

    if not image_paths:
        raise RuntimeError("Slideshow images were found but none could be downloaded.")

    return image_paths


# ── Instagram Reels/posts (Apify) ─────────────────────────────────────────────
#
# Uses the official "Instagram Scraper" actor (apify/instagram-scraper) —
# a single call returns BOTH the video content (caption, direct videoUrl,
# owner) AND the post's top comments (latestComments), unlike TikTok which
# needs two separate actors for the same two things. Input and output
# fields below were confirmed against a real run (Apify Console → Runs →
# Input/Output → JSON), not guessed:
#
#   Input: {"directUrls": [url], "resultsType": "reels", "resultsLimit": 1,
#           "addParentData": false}
#
#   Output (one dataset item per post): caption, videoUrl, displayUrl,
#     ownerUsername, ownerFullName, shortCode, timestamp, likesCount,
#     videoViewCount, commentsCount, isCommentsDisabled, locationName,
#     latestComments (list of {text, ownerUsername, likesCount, timestamp,
#     ownerProfilePicUrl, repliesCount})
#
# Instagram's own bot detection is aggressive against datacenter IPs (the
# same problem that killed YouTube support), but this actor runs on
# Apify's managed residential proxies rather than Railway's own IP, so it
# sidesteps that specific issue — same pattern that already works for the
# TikTok slideshow scraper above.

APIFY_INSTAGRAM_ACTOR = "apify~instagram-scraper"


def fetch_instagram_post(url: str) -> dict:
    """
    Fetches one Instagram Reel/post's video + caption + top comments in a
    single Apify call.

    Returns a dict with keys: title, description (both used by the troll
    filter, same as fetch_metadata/fetch_slideshow_post), video_url,
    uploader, comments (list of dicts matching fetch_top_comments' shape:
    text, username, likes, reply_count, avatar_url, created_at).

    Raises RuntimeError on failure — no video (e.g. it's a photo-only
    post), private/deleted post, or the Apify call itself failing.
    """
    if not APIFY_API_TOKEN:
        raise RuntimeError(
            "APIFY_API_TOKEN is not set — add it in Railway → Variables."
        )

    endpoint = f"https://api.apify.com/v2/acts/{APIFY_INSTAGRAM_ACTOR}/run-sync-get-dataset-items"
    payload = {
        "directUrls": [url],
        "resultsType": "reels",
        "resultsLimit": 1,
        "addParentData": False,
    }

    try:
        resp = requests.post(
            endpoint, params={"token": APIFY_API_TOKEN}, json=payload, timeout=60,
        )
        resp.raise_for_status()
        items = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Apify Instagram request failed: {e}")

    if not items:
        raise RuntimeError(
            "No data returned for this Instagram post — it may be private, "
            "deleted, or the link isn't a public Reel."
        )

    post = items[0]

    video_url = post.get("videoUrl", "")
    if not video_url:
        raise RuntimeError(
            "This Instagram post has no video attached — is it a photo "
            "carousel rather than a Reel? Photo-only Instagram posts "
            "aren't supported yet."
        )

    caption = (post.get("caption") or "").strip()

    comments = []
    for c in (post.get("latestComments") or []):
        text = (c.get("text") or "").strip()
        if not text:
            continue
        comments.append({
            "text": text,
            "username": c.get("ownerUsername", ""),
            "likes": c.get("likesCount", 0) or 0,
            "reply_count": c.get("repliesCount") or 0,
            "avatar_url": c.get("ownerProfilePicUrl", ""),
            "created_at": c.get("timestamp", ""),
        })
    comments.sort(key=lambda c: c["likes"], reverse=True)

    return {
        "title": caption[:200],
        "description": caption,
        "video_url": video_url,
        "uploader": post.get("ownerUsername", ""),
        "comments": comments,
    }


def download_instagram_video(video_url: str, output_dir: str) -> str:
    """
    Downloads the direct Instagram CDN video URL from fetch_instagram_post().
    Unlike TikTok, no yt-dlp is needed here — Apify already resolved the
    real playable MP4 URL, so it's a plain HTTP download.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GetWayBot/1.0)",
        "Referer": "https://www.instagram.com/",
    }
    try:
        resp = requests.get(video_url, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Instagram video download failed: {e}")

    out_path = os.path.join(output_dir, "instagram_video.mp4")
    with open(out_path, "wb") as f:
        f.write(resp.content)
    return out_path


# ── Real TikTok comments (Apify) ──────────────────────────────────────────────
#
# Uses the "TikTok Comments Scraper" actor (clockworks/tiktok-comments-scraper)
# — same vendor as the slideshow scraper above. Both the input parameter
# names and the output field names below were confirmed against a real run
# (Apify Console → Runs → Output → JSON), not guessed:
#
#   Input (apify.com/clockworks/tiktok-comments-scraper/input-schema):
#     postURLs, commentsPerPost, topLevelCommentsPerPost, maxRepliesPerComment
#
#   Output (one dataset item per comment):
#     text, diggCount, replyCommentTotal, createTimeISO, uniqueId, avatarThumbnail
#
# Comments are a nice-to-have, not core to the itinerary — callers should
# treat failures here as non-fatal and just skip the comments section
# rather than blocking route generation.

APIFY_COMMENTS_ACTOR = "clockworks~tiktok-comments-scraper"


def fetch_top_comments(url: str, max_comments: int = 15) -> list[dict]:
    """
    Fetches up to `max_comments` top-level comments for a TikTok post,
    sorted by like count (most-liked first). Returns a list of dicts with
    keys: text, username, likes, reply_count, avatar_url, created_at.
    """
    if not APIFY_API_TOKEN:
        raise RuntimeError(
            "APIFY_API_TOKEN is not set — add it in Railway → Variables."
        )

    endpoint = f"https://api.apify.com/v2/acts/{APIFY_COMMENTS_ACTOR}/run-sync-get-dataset-items"
    payload = {
        "postURLs": [url],
        "commentsPerPost": max_comments,
        "topLevelCommentsPerPost": max_comments,
        "maxRepliesPerComment": 0,  # replies aren't shown in the UI — skip them to save cost
    }

    try:
        resp = requests.post(
            endpoint, params={"token": APIFY_API_TOKEN}, json=payload, timeout=60,
        )
        resp.raise_for_status()
        items = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Apify comments request failed: {e}")

    comments = []
    for item in items:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        comments.append({
            "text": text,
            "username": item.get("uniqueId", ""),
            "likes": item.get("diggCount", 0) or 0,
            "reply_count": item.get("replyCommentTotal", 0) or 0,
            "avatar_url": item.get("avatarThumbnail", ""),
            "created_at": item.get("createTimeISO", ""),
        })

    # Most-liked first — surfaces the comments people actually cared about.
    comments.sort(key=lambda c: c["likes"], reverse=True)
    return comments[:max_comments]


# ── Frame extraction — OpenCV only, zero system deps ─────────────────────────

def extract_frames(video_path: str, output_dir: str, n_frames: int = 8) -> list[str]:
    """
    Extracts n_frames evenly-spaced JPEG screenshots using OpenCV (cv2).

    Why cv2 instead of ffmpeg:
      - Pure pip install, no system packages, works on Mac/Linux/Windows
      - cv2.VideoCapture reads mp4 natively via its own bundled decoder
      - Produces identical quality output to ffmpeg for JPEG frames

    Skips first 3% and last 3% to avoid intro/outro cards.
    Resizes to 960px wide to balance quality vs Claude API token cost.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV could not open the video file: {video_path}\n"
            "Make sure the video downloaded correctly and is a valid mp4."
        )

    # Get video properties via cv2 (no ffprobe needed)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    if duration_sec < 3:
        cap.release()
        raise RuntimeError("Video is too short to extract meaningful frames.")

    # Build evenly-spaced timestamps, skipping 3% intro + 3% outro
    start_sec = duration_sec * 0.03
    end_sec   = duration_sec * 0.97
    span      = end_sec - start_sec
    step      = span / max(n_frames - 1, 1)
    timestamps = [start_sec + i * step for i in range(n_frames)]

    frame_paths: list[str] = []

    for i, ts in enumerate(timestamps):
        # Seek to the exact timestamp (milliseconds for cv2)
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()

        if not ret or frame is None:
            # Some encodings miss a frame near the end — skip silently
            continue

        # Resize to 960px wide, preserving aspect ratio
        h, w = frame.shape[:2]
        if w > 960:
            scale  = 960 / w
            new_w  = 960
            new_h  = int(h * scale)
            frame  = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        out_path = os.path.join(output_dir, f"frame_{i:02d}.jpg")
        # JPEG quality 85 — good visual clarity without excess token cost
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_paths.append(out_path)

    cap.release()

    if not frame_paths:
        raise RuntimeError(
            "OpenCV extracted zero frames. The video may be corrupted or "
            "in a format cv2 cannot decode on this system."
        )

    return frame_paths
