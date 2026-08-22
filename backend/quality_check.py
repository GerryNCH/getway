"""
quality_check.py — AI Quality Check for generated itineraries.

Runs a QA pass over a generated Itinerary BEFORE it reaches the admin's
review queue: flags stops with generic/unnamed places, missing or invalid
coordinates, and missing photos.

IMPORTANT: this only flags problems — it never approves, rejects, or
publishes anything. Final approval is always a manual decision in the
admin panel (POST /admin/approve/{video_id}).

By design this does NOT check:
  - Stops per day: a day can legitimately be a single full-day activity
    (a hike, a boat trip, a long museum visit) — stop count alone isn't a
    quality signal.
  - Hotel presence: GetWay auto-populates a generic "Hotels in [city]"
    fallback elsewhere in the product for routes with no hotel stop, so an
    empty hotel_banner_photo_url on the Itinerary object doesn't mean the
    visitor sees no hotel content.

Split between two layers, on purpose:
  - Countable facts (coordinate RANGE validity, placeholder/missing
    photos) are checked in plain Python — free, exact, instant. No reason
    to spend a model call on arithmetic.
  - The one genuine judgment call (is this name a real, specific,
    bookable place, or a vague placeholder?) goes to Claude Haiku
    (claude-haiku-4-5-20251001) — cheap (~$0.001/check) and fast.

This deliberately does NOT ask Haiku to judge whether coordinates
plausibly belong to the destination's country/region. That check existed
in an earlier version and was removed: Haiku has no real geocoding
database, only its own approximate world knowledge, and for smaller or
less-famous places (a village in Malta, a specific cathedral) it produced
confident-sounding but wrong "coordinates don't match" flags — worse than
not checking at all, since a wrong flag costs the admin's trust and time
investigating a non-issue. Genuinely broken coordinates (out of the
-90..90 / -180..180 range, or missing entirely) are still caught, for
free, by the deterministic layer below.

Both layers report per-stop, and are merged into ONE issue line per
problematic stop (e.g. "generic name, missing photo") rather than one
line per sub-problem — keeps the admin panel readable instead of
repeating the same stop three times.

A Haiku failure (rate limit, bad JSON, network error, etc.) is non-fatal:
the deterministic checks still run and are returned, since a partial
quality check is far more useful to the admin than none at all.
"""

import json
import re

import anthropic
from models import Itinerary

_client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env

# Claude Haiku 4.5 standard rate, $ per million tokens (verified July 2026 —
# same figures as troll_filter.py; update both if pricing changes or the
# model string below is swapped for a different one).
_HAIKU_INPUT_PER_MTOK = 1.00
_HAIKU_OUTPUT_PER_MTOK = 5.00

# Known placeholder-photo domains — a stop photo pointing here is the same
# as having no real photo at all (see index.html's old Lorem Picsum demo
# fallbacks).
_PLACEHOLDER_PHOTO_MARKERS = ("picsum.photos",)

_QUALITY_SYSTEM = """You are a travel-itinerary QA reviewer for GetWay, a TikTok-to-itinerary travel platform.

You will receive a destination and a flat list of stops (day, stop_number, name, category). Judge ONE thing only:

GENERIC NAMES — does `name` identify one real, specific place, OR a real specific village/town/neighbourhood/district that is itself the intended stop (e.g. "explore this village")? Be conservative — when in doubt, do NOT flag it.

Reject only clear, vague placeholders that describe a TYPE of place instead of naming one: "Restaurant near the Colosseum", "Hotel in Rome", "Unknown location", "Local cafe", "A beach in the south", "Cafe in Montmartre", "Cosy dinner restaurant, St. Paul's Bay", "Place near city".

Do NOT flag:
- Real town/village/neighbourhood names, even paired with just the country/region — e.g. "Qrendi, Malta", "Żurrieq, Malta", "Trastevere, Rome" are real places and a legitimate whole-place stop, not placeholders.
- Real named businesses, attractions, or landmarks, even ones you personally don't recognise — e.g. "The Limestone Heritage, Valletta", "Ristorante Aroma", "Cala Llombards" all name one specific place.
- Anything that reads as a proper noun / capitalized specific name, as opposed to a generic noun phrase.

If you are not highly confident a name is a vague category description rather than an actual specific place, leave it out.

Reply with ONLY a JSON object, no preamble, no markdown fences:
{
  "generic_names": [{"day": 1, "stop_number": 1, "name": "...", "suggestion": "one short sentence in English on what to fix"}]
}

The array can be empty. day/stop_number must exactly match the input. All "suggestion" text must be written in English."""


def _is_placeholder_photo(url: str) -> bool:
    """True for an empty photo_url or a known placeholder-image domain."""
    if not url or not url.strip():
        return True
    low = url.lower()
    return any(marker in low for marker in _PLACEHOLDER_PHOTO_MARKERS)


def _deterministic_stop_findings(itinerary: Itinerary) -> dict:
    """
    Per-stop coordinate/photo problems — everything countable without a
    model call. Keyed by (day, stop_number).

    Returns {(day, stop_number): {"name": str, "problems": [str], "deduction": int}}
    """
    findings: dict = {}

    for day in itinerary.days:
        for i, stop in enumerate(day.stops, start=1):
            problems = []
            deduction = 0

            if stop.lat is None or stop.lng is None:
                problems.append("missing coordinates")
                deduction += 3
            elif not (-90 <= stop.lat <= 90) or not (-180 <= stop.lng <= 180):
                problems.append("invalid coordinates")
                deduction += 10

            if _is_placeholder_photo(stop.photo_url):
                problems.append("missing photo")
                deduction += 5

            if problems:
                findings[(day.day, i)] = {
                    "name": stop.name,
                    "problems": problems,
                    "deduction": deduction,
                }

    return findings


def _haiku_stop_findings(itinerary: Itinerary) -> tuple[dict, float]:
    """
    Claude Haiku's judgment on generic vs. real specific stop names — the
    one genuine judgment call in this file. Keyed by (day, stop_number),
    same shape as _deterministic_stop_findings, plus an optional
    "suggestion" key.

    Returns (findings, cost_usd). Never raises — on any failure (bad JSON,
    network, rate limit) returns ({}, 0.0) so the deterministic findings
    above still get returned on their own.
    """
    stops_payload = []
    for day in itinerary.days:
        for i, stop in enumerate(day.stops, start=1):
            stops_payload.append({
                "day": day.day,
                "stop_number": i,
                "name": stop.name,
                "category": stop.category,
            })

    user_content = json.dumps({
        "destination": itinerary.destination,
        "stops": stops_payload,
    }, ensure_ascii=False)

    findings: dict = {}
    cost_usd = 0.0

    try:
        response = _client.messages.create(
            model="claude-haiku-4-5-20251001",   # cheapest/fastest model — this is a QA pass, not the main analysis
            max_tokens=1200,
            system=_QUALITY_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
        )
        usage = getattr(response, "usage", None)
        if usage:
            cost_usd = (
                usage.input_tokens * _HAIKU_INPUT_PER_MTOK
                + usage.output_tokens * _HAIKU_OUTPUT_PER_MTOK
            ) / 1_000_000

        raw = response.content[0].text.strip()
        # Defensive: strip stray markdown fences if the model adds them
        # despite being told not to.
        if raw.startswith("```"):
            raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(raw)

        for item in result.get("generic_names", []):
            key = (item.get("day"), item.get("stop_number"))
            entry = findings.setdefault(key, {"name": item.get("name", ""), "problems": [], "deduction": 0})
            entry["problems"].append("generic name")
            entry["deduction"] += 15
            if item.get("suggestion"):
                entry["suggestion"] = item["suggestion"]

    except Exception as e:
        # A failed QA pass must never block route generation — just log it
        # and fall back to the deterministic checks only.
        print(f"[QualityCheck] Haiku semantic check failed (non-fatal): {e}")

    return findings, cost_usd


def ai_quality_check(itinerary: Itinerary) -> dict:
    """
    Runs the full AI quality check on a generated itinerary and returns a
    JSON-serializable result:

        {
          "score": int,          # 0-100
          "status": str,         # "good" | "needs_attention" | "poor"
          "issues": [str, ...],
          "suggestions": [str, ...],
          "cost_usd": float,
        }

    Status thresholds: good 80-100, needs_attention 50-79, poor under 50.

    Each problematic stop produces exactly ONE issue line combining every
    sub-problem found for it (e.g. "generic name, missing photo") — kept
    compact on purpose so the admin panel doesn't repeat the same stop
    three times over.

    Never raises. This is a QA aid for the admin, not a gate — the itinerary
    is generated (or shown in the panel) exactly the same whether this
    check succeeds, partially succeeds, or fails outright.
    """
    try:
        det_findings = _deterministic_stop_findings(itinerary)
        haiku_findings, cost_usd = _haiku_stop_findings(itinerary)

        # Merge the two per-stop dicts.
        merged: dict = {}
        for key, entry in det_findings.items():
            merged[key] = dict(entry)
        for key, entry in haiku_findings.items():
            if key in merged:
                merged[key]["problems"] += entry["problems"]
                merged[key]["deduction"] += entry["deduction"]
                if "suggestion" in entry:
                    merged[key]["suggestion"] = entry["suggestion"]
            else:
                merged[key] = entry

        issues: list[str] = []
        suggestions: list[str] = []
        deduction = 0

        for (day, stop_num), entry in sorted(merged.items()):
            deduction += entry["deduction"]
            issues.append(f"Day {day}, Stop {stop_num}: '{entry['name']}' — {', '.join(entry['problems'])}")
            if entry.get("suggestion"):
                suggestions.append(entry["suggestion"])
            else:
                suggestions.append(f"Fix '{entry['name']}': {', '.join(entry['problems'])}")

        score = max(0, min(100, 100 - deduction))

        if score >= 80:
            status = "good"
        elif score >= 50:
            status = "needs_attention"
        else:
            status = "poor"

        return {
            "score": score,
            "status": status,
            "issues": issues,
            "suggestions": suggestions,
            "cost_usd": round(cost_usd, 6),
        }
    except Exception as e:
        print(f"[QualityCheck] Unexpected error: {e}")
        return {
            "score": 0,
            "status": "poor",
            "issues": [f"AI quality check failed to run: {e}"],
            "suggestions": ["Check the route manually — the automatic check crashed."],
            "cost_usd": 0.0,
        }
