"""
quality_check.py — AI Quality Check for generated itineraries.

Runs a QA pass over a generated Itinerary BEFORE it reaches the admin's
review queue: flags generic/unnamed stops, thin days, missing hotel
content, broken coordinates, and missing photos.

IMPORTANT: this only flags problems — it never approves, rejects, or
publishes anything. Final approval is always a manual decision in the
admin panel (POST /admin/approve/{video_id}).

Split between two layers, on purpose:
  - Countable facts (stop count/day, hotel presence, coordinate RANGE
    validity, placeholder/missing photos) are checked in plain Python —
    free, exact, instant. No reason to spend a model call on arithmetic.
  - Judgment calls (is this name a real specific place? do these
    coordinates plausibly belong to this destination's country?) go to
    Claude Haiku (claude-haiku-4-5-20251001) — cheap (~$0.001/check) and
    fast, and this is exactly the kind of fuzzy judgment Haiku is good at.

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

You will receive a destination and a flat list of stops (day, stop_number, name, category, lat, lng). Judge ONLY two things:

1. GENERIC NAMES — does `name` identify one real, specific, bookable place? Reject vague placeholders such as "Restaurant near the Colosseum", "Hotel in Rome", "Unknown location", "Local cafe", "A beach in the south", "Place near city". Accept real proper names even if you cannot personally confirm they exist (e.g. "Ristorante Aroma", "Hotel Artemide", "Cala Llombards").

2. COORDINATE PLAUSIBILITY — does (lat, lng) fall roughly within the country/region implied by the destination? Only flag stops that are CLEARLY wrong (e.g. a Rome stop with coordinates in another continent) — not small in-city imprecision. Skip stops where lat or lng is null.

Reply with ONLY a JSON object, no preamble, no markdown fences:
{
  "generic_names": [{"day": 1, "stop_number": 1, "name": "...", "suggestion": "one short sentence in Bulgarian on what to fix"}],
  "coordinate_issues": [{"day": 1, "stop_number": 1, "name": "...", "suggestion": "one short sentence in Bulgarian on what to fix"}]
}

Both arrays can be empty. day/stop_number must exactly match the input. All "suggestion" text must be written in Bulgarian."""


def _is_placeholder_photo(url: str) -> bool:
    """True for an empty photo_url or a known placeholder-image domain."""
    if not url or not url.strip():
        return True
    low = url.lower()
    return any(marker in low for marker in _PLACEHOLDER_PHOTO_MARKERS)


def _deterministic_checks(itinerary: Itinerary) -> tuple[list[str], list[str], int]:
    """
    Everything countable without a model call: stops/day, hotel presence,
    coordinate range validity, and placeholder/missing photos.

    Returns (issues, suggestions, score_deduction).
    """
    issues: list[str] = []
    suggestions: list[str] = []
    deduction = 0

    # A hotel stop OR an admin-set "Hotels in [city]" fallback banner both
    # satisfy "at least 1 hotel stop or hotel banner" per the spec.
    has_hotel = bool(itinerary.hotel_banner_photo_url)
    missing_or_invalid_coords = 0
    missing_photos = 0

    for day in itinerary.days:
        if len(day.stops) < 3:
            issues.append(f"Day {day.day}: Само {len(day.stops)} спирки (нужни поне 3)")
            deduction += 10

        for i, stop in enumerate(day.stops, start=1):
            if stop.category == "hotel":
                has_hotel = True

            if stop.lat is None or stop.lng is None:
                missing_or_invalid_coords += 1
                issues.append(f"Day {day.day}, Stop {i}: Липсват координати за '{stop.name}'")
                deduction += 3
            elif not (-90 <= stop.lat <= 90) or not (-180 <= stop.lng <= 180):
                missing_or_invalid_coords += 1
                issues.append(f"Day {day.day}, Stop {i}: Невалидни координати за '{stop.name}'")
                deduction += 10

            if _is_placeholder_photo(stop.photo_url):
                missing_photos += 1
                issues.append(f"Day {day.day}, Stop {i}: Липсва снимка за спирка '{stop.name}'")
                deduction += 5

    if not has_hotel:
        issues.append("Липсва хотел спирка или hotel banner снимка за маршрута")
        suggestions.append("Добави хотел спирка или качи hotel banner снимка за маршрута")
        deduction += 15

    if missing_or_invalid_coords:
        suggestions.append("Провери и коригирай координатите на маркираните спирки")
    if missing_photos:
        suggestions.append("Добави липсващи снимки за маркираните спирки")

    return issues, suggestions, deduction


def _haiku_semantic_checks(itinerary: Itinerary) -> tuple[list[str], list[str], int, float]:
    """
    Claude Haiku's judgment on generic names and coordinate/country
    plausibility — the two checks that need real reasoning, not just
    arithmetic.

    Returns (issues, suggestions, score_deduction, cost_usd). Never raises —
    on any failure (bad JSON, network, rate limit) returns an all-empty,
    zero-cost result so the deterministic checks above still get returned.
    """
    stops_payload = []
    for day in itinerary.days:
        for i, stop in enumerate(day.stops, start=1):
            stops_payload.append({
                "day": day.day,
                "stop_number": i,
                "name": stop.name,
                "category": stop.category,
                "lat": stop.lat,
                "lng": stop.lng,
            })

    user_content = json.dumps({
        "destination": itinerary.destination,
        "stops": stops_payload,
    }, ensure_ascii=False)

    issues: list[str] = []
    suggestions: list[str] = []
    deduction = 0
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
            day, num = item.get("day"), item.get("stop_number")
            name = item.get("name", "")
            issues.append(f"Day {day}, Stop {num}: Неконкретно име — '{name}'")
            deduction += 15
            if item.get("suggestion"):
                suggestions.append(item["suggestion"])

        for item in result.get("coordinate_issues", []):
            day, num = item.get("day"), item.get("stop_number")
            name = item.get("name", "")
            issues.append(f"Day {day}, Stop {num}: Координатите не отговарят на дестинацията — '{name}'")
            deduction += 10
            if item.get("suggestion"):
                suggestions.append(item["suggestion"])

    except Exception as e:
        # A failed QA pass must never block route generation — just log it
        # and fall back to the deterministic checks only.
        print(f"[QualityCheck] Haiku semantic check failed (non-fatal): {e}")

    return issues, suggestions, deduction, cost_usd


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

    Never raises. This is a QA aid for the admin, not a gate — the itinerary
    is generated (or shown in the panel) exactly the same whether this
    check succeeds, partially succeeds, or fails outright.
    """
    try:
        det_issues, det_suggestions, det_deduction = _deterministic_checks(itinerary)
        sem_issues, sem_suggestions, sem_deduction, cost_usd = _haiku_semantic_checks(itinerary)

        issues = det_issues + sem_issues
        suggestions = det_suggestions + sem_suggestions
        score = max(0, min(100, 100 - det_deduction - sem_deduction))

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
            "issues": [f"AI quality check не успя да се изпълни: {e}"],
            "suggestions": ["Провери маршрута ръчно — автоматичната проверка гръмна."],
            "cost_usd": 0.0,
        }
