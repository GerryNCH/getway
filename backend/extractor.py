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


# ── Slideshow image download ─────────────────────────────────────────────────

def download_slideshow_images(url: str, output_dir: str) -> list[str]:
    """
    Downloads the individual images from a TikTok slideshow (/photo/) post.

    TikTok slideshows have no video stream, so yt-dlp can't download them the
    normal way. Instead we pull the post's metadata JSON (same --dump-json
    call used for fetch_metadata) and read the slide images straight out of
    it, then fetch each one directly:

      - Newer yt-dlp versions expose the slides under `image_post_info.images`
        (each with a `display_image.url_list` of CDN URLs).
      - As a fallback, yt-dlp always normalises extractor images into the
        generic `thumbnails` list, which for a slideshow post is the set of
        slides themselves.

    Returns paths to the downloaded slide images, in original slide order —
    these get passed to analyse_frames() exactly like video frames would.
    """
    result = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-download",
         "--no-playlist", "--quiet", url],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not fetch slideshow metadata: {result.stderr.strip()}"
        )

    data = json.loads(result.stdout)

    image_urls: list[str] = []

    image_post = data.get("image_post_info") or {}
    for img in image_post.get("images", []):
        url_list = (img.get("display_image") or {}).get("url_list") or []
        if url_list:
            image_urls.append(url_list[0])

    if not image_urls:
        seen = set()
        for thumb in data.get("thumbnails", []):
            thumb_url = thumb.get("url")
            if thumb_url and thumb_url not in seen:
                seen.add(thumb_url)
                image_urls.append(thumb_url)

    if not image_urls:
        raise RuntimeError(
            "No slideshow images were found in this TikTok post's metadata."
        )

    headers = {"User-Agent": "Mozilla/5.0 (compatible; GetWayBot/1.0)"}
    image_paths: list[str] = []

    for i, img_url in enumerate(image_urls):
        try:
            resp = requests.get(img_url, headers=headers, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[Slideshow] Skipping image {i} — download failed: {e}")
            continue

        out_path = os.path.join(output_dir, f"slide_{i:02d}.jpg")
        with open(out_path, "wb") as f:
            f.write(resp.content)
        image_paths.append(out_path)

    if not image_paths:
        raise RuntimeError(
            "Slideshow images were found but none could be downloaded."
        )

    return image_paths


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
