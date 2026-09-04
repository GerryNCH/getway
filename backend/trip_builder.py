"""
trip_builder.py — "Build Your Own Trip" candidate curation (Phase A).

Turns a broad Google Places attraction search (places.search_attractions_broad)
into a clean, budget-appropriate candidate list for a destination:
  1. Filter out closed places (businessStatus) — is_open()
  2. Classify each place into which budget tier(s) it fits — paid places by
     priceLevel, free places (parks/plazas/outdoor monuments) by type, and
     the most famous places (by rating x review count) always included
     regardless of tier — fits_budget()
  3. AI curation pass (Claude Haiku) over the filtered raw results — drops
     junk/duplicates, rewrites descriptions in GetWay's site tone —
     curate_candidates()

Same spirit as quality_check.py: deterministic/countable filtering stays in
plain Python (free, exact, instant); the one genuine judgment call (which
raw Places results are actually worth showing, and how to describe them)
goes to Haiku.

Deliberately does NOT touch the database — main.py owns the (city, budget)
cache read/write, same division of responsibility as quality_check.py
(this module is pure logic; main.py wires it to storage).
"""

import json

import anthropic
from places import _build_photo_url

_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

# Claude Haiku 4.5 standard rate, $ per million tokens — same figures used
# in quality_check.py and troll_filter.py.
_HAIKU_INPUT_PER_MTOK = 1.00
_HAIKU_OUTPUT_PER_MTOK = 5.00

# Places API (New) businessStatus values. Missing entirely (some results
# don't set it) is treated as open — Google's own default assumption.
_CLOSED_STATUSES = {"CLOSED_TEMPORARILY", "CLOSED_PERMANENTLY"}

# Place `types` that indicate a free-to-enter outdoor attraction — used to
# classify a place as budget-tier-agnostic ("free") when it has no
# priceLevel at all, which is normal for this category (Google's
# priceLevel field is really meant for restaurants/bars, not parks).
_FREE_TYPE_HINTS = {
    "park", "plaza", "national_park", "state_park", "hiking_area",
    "historical_landmark", "monument", "natural_feature", "beach",
    "square", "garden", "botanical_garden", "observation_deck",
}

# Which Places (New) priceLevel enum values are appropriate for each budget
# tier. Free places are handled separately by type (_FREE_TYPE_HINTS) —
# relying on priceLevel alone would starve every tier, since most
# attractions never set it.
_PRICE_LEVEL_BY_TIER = {
    "cheap": {"PRICE_LEVEL_FREE", "PRICE_LEVEL_INEXPENSIVE"},
    "mid": {"PRICE_LEVEL_FREE", "PRICE_LEVEL_INEXPENSIVE", "PRICE_LEVEL_MODERATE"},
    "luxury": {"PRICE_LEVEL_FREE", "PRICE_LEVEL_MODERATE", "PRICE_LEVEL_EXPENSIVE", "PRICE_LEVEL_VERY_EXPENSIVE"},
}

# A place this well-established is shown regardless of budget tier — e.g.
# the Eiffel Tower belongs on a "cheap" trip's candidate list too, even
# though it has no useful priceLevel signal either way.
_FAMOUS_MIN_RATING = 4.5
_FAMOUS_MIN_REVIEWS = 2000


def is_open(place: dict) -> bool:
    """True unless Places explicitly reports this place as closed."""
    return place.get("businessStatus", "") not in _CLOSED_STATUSES


def _is_famous(place: dict) -> bool:
    rating = place.get("rating", 0) or 0
    reviews = place.get("userRatingCount", 0) or 0
    return rating >= _FAMOUS_MIN_RATING and reviews >= _FAMOUS_MIN_REVIEWS


def _is_free_by_type(place: dict) -> bool:
    types = set(place.get("types", []) or [])
    return bool(types & _FREE_TYPE_HINTS) and not place.get("priceLevel")


def fits_budget(place: dict, budget: str) -> bool:
    """
    True if `place` (a raw Places API (New) result — see
    places.search_attractions_broad) belongs on the candidate list for
    `budget` ("cheap" | "mid" | "luxury").
    """
    if _is_famous(place):
        return True
    if _is_free_by_type(place):
        return True
    price_level = place.get("priceLevel", "")
    if not price_level:
        # No price signal, not free-by-type, not famous — most likely a
        # paid attraction Google just hasn't priced. Only surface it on
        # "mid" (the safe middle default) rather than guessing wrong on
        # "cheap" or "luxury".
        return budget == "mid"
    return price_level in _PRICE_LEVEL_BY_TIER.get(budget, set())


def _place_to_candidate_dict(place: dict, description: str = "", category: str = "sight") -> dict:
    """Converts one raw Places result into a TripCandidate-shaped dict."""
    loc = place.get("location") or {}
    photos = place.get("photos", [])
    photo_url = ""
    if photos:
        landscape = [p for p in photos if p.get("widthPx", 0) > p.get("heightPx", 0)]
        best = max(landscape or photos, key=lambda p: p.get("widthPx", 0))
        photo_url = _build_photo_url(best.get("name", ""))
    name = place.get("displayName", {}).get("text", "")
    return {
        "name": name,
        "description": description or name,
        "category": category,
        "photo_url": photo_url,
        "rating": place.get("rating", 0) or 0,
        "user_rating_count": place.get("userRatingCount", 0) or 0,
        "price_level": place.get("priceLevel", "") or "",
        "lat": loc.get("latitude"),
        "lng": loc.get("longitude"),
        "is_famous": _is_famous(place),
    }


_CURATION_SYSTEM = """You are curating raw Google Places search results into a clean candidate-attraction list for GetWay, a travel itinerary app.

You will receive a destination and a JSON array of raw candidates, each with an "index", "name", "types", "rating", and "user_rating_count".

Your job for EACH candidate:
1. Decide whether it's a genuine, tourist-worthy attraction, activity, or place to eat — DROP anything that clearly isn't (parking garages, generic ATMs/banks, gas stations, residential buildings, chain pharmacies, anything that isn't actually a place a traveler would choose to visit).
2. DROP near-duplicates of another candidate in the same list (e.g. two listings for the same landmark under slightly different names) — keep only the better-named one.
3. For everything you keep, write ONE short, engaging sentence description in GetWay's tone — like a travel writer's pick, not a dry Google Maps category label. Never invent specific facts you can't reasonably infer from the name/type.
4. Assign one category: sight | food | activity | beach | village. Never "hotel" — this list never includes accommodation.

Reply with ONLY valid JSON, no markdown fences:
{"candidates": [{"index": 0, "description": "...", "category": "sight"}]}

Only include entries you decided to KEEP — dropped candidates simply don't appear in the array. "index" must exactly match an index from the input."""


def curate_candidates(destination: str, raw_places: list[dict]) -> tuple[list[dict], float]:
    """
    Runs the AI curation pass over `raw_places` (already filtered for
    open/budget-fit — see is_open() and fits_budget() above) and returns
    (candidates, cost_usd), where each candidate is a plain dict ready for
    models.TripCandidate(**candidate).

    Never raises: on any AI failure this falls back to returning every
    input place unfiltered, with its own Places displayName standing in
    for a description — a slightly rougher candidate list beats a broken
    endpoint (same non-fatal philosophy as quality_check.ai_quality_check).
    """
    if not raw_places:
        return [], 0.0

    payload_places = [
        {
            "index": i,
            "name": place.get("displayName", {}).get("text", ""),
            "types": place.get("types", []),
            "rating": place.get("rating", 0),
            "user_rating_count": place.get("userRatingCount", 0),
        }
        for i, place in enumerate(raw_places)
    ]

    cost_usd = 0.0
    try:
        response = _client.messages.create(
            model="claude-haiku-4-5-20251001",  # cheapest/fastest — this is a curation pass, not the main analysis
            max_tokens=4000,
            system=_CURATION_SYSTEM,
            messages=[{"role": "user", "content": json.dumps({
                "destination": destination,
                "candidates": payload_places,
            }, ensure_ascii=False)}],
        )
        usage = getattr(response, "usage", None)
        if usage:
            cost_usd = (
                usage.input_tokens * _HAIKU_INPUT_PER_MTOK
                + usage.output_tokens * _HAIKU_OUTPUT_PER_MTOK
            ) / 1_000_000

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(raw)

        candidates = []
        for item in result.get("candidates", []):
            idx = item.get("index")
            if idx is None or not (0 <= idx < len(raw_places)):
                continue
            candidates.append(_place_to_candidate_dict(
                raw_places[idx],
                description=item.get("description", ""),
                category=item.get("category", "sight"),
            ))
        return candidates, cost_usd

    except Exception as e:
        print(f"[TripBuilder] AI curation failed (non-fatal, returning unfiltered list): {type(e).__name__}: {e}")
        return [_place_to_candidate_dict(p) for p in raw_places], cost_usd
