"""
trip_builder.py — "Build Your Own Trip" candidate curation (Phase A) and
hotel selection (Phase B).

Phase A turns a broad Google Places attraction search
(places.search_attractions_broad) into a clean, budget-appropriate
candidate list for a destination:
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

Phase B picks a single recommended hotel, geographically anchored to the
traveler's selected attractions:
  4. cluster_center() — centroid of the selected attractions' coordinates
  5. pick_hotel() — from places.search_hotels_near() results biased toward
     that centroid, picks the best budget-tier fit among hotels that are
     open AND 4.0+ rated — LOCKED LOGIC: "cheap" only ever changes which
     room-rate tier gets picked, never location quality or star rating.
  6. hotel_to_recommendation_dict() — shapes the chosen hotel into a
     TripHotelRecommendation-ready dict, with real Booking.com/Expedia
     affiliate links built via the SAME functions ai_analyzer.py already
     uses for video-extracted hotel stops (no new affiliate-link logic).

Phase C assembles the actual day-by-day itinerary from the traveler's
selections, in the SAME Stop/DayPlan/Itinerary shape the video-generated
routes already use:
  7. group_into_days() — one greedy nearest-neighbor path across ALL
     selected attractions (_nearest_neighbor_order), split into `days`
     contiguous chunks. A contiguous slice of an already-proximity-ordered
     path is naturally a geographic cluster, and day N+1 starts near where
     day N ended — minimizes backtracking both within a day and across the
     whole trip without a full routing-optimization solve.
  8. assemble_days() — turns those day-clusters (+ the Phase B hotel) into
     DayPlan-shaped dicts of Stop-shaped dicts.
  9. recommend_car_rental() — a SEPARATE Claude Haiku call reasoning from
     real-world knowledge of the destination itself (driving conditions,
     transit quality, parking) — deliberately NOT a geometric heuristic
     based on how spread out the selected stops are.

Deliberately does NOT touch the database — main.py owns the (city, budget[,
location]) cache read/write, same division of responsibility as
quality_check.py (this module is pure logic; main.py wires it to storage).
"""

import json
import math
import statistics

import anthropic
from places import photo_url_from_places_photos
from ai_analyzer import _booking_affiliate_url, _expedia_affiliate_url

_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

# Claude Haiku 4.5 standard rate, $ per million tokens — same figures used
# in quality_check.py and troll_filter.py.
_HAIKU_INPUT_PER_MTOK = 1.00
_HAIKU_OUTPUT_PER_MTOK = 5.00


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance between two (lat, lng) points, in km. Shared
    by pick_hotel's distance sanity check (Phase B) and the day-clustering
    path-ordering (Phase C)."""
    lat1, lng1 = math.radians(a[0]), math.radians(a[1])
    lat2, lng2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlng = lat2 - lat1, lng2 - lng1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))

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
    photo_url = photo_url_from_places_photos(place.get("photos", []))
    name = place.get("displayName", {}).get("text", "")
    return {
        "name": name,
        "description": description or name,
        "category": category,
        "photo_url": photo_url,
        "rating": place.get("rating", 0) or 0,
        "user_rating_count": place.get("userRatingCount", 0) or 0,
        "price_level": place.get("priceLevel", "") or "",
        "is_free": _is_free_by_type(place),
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


# ── Phase B: hotel selection ─────────────────────────────────────────────

# LOCKED LOGIC: a hotel must be this well-rated no matter the budget tier —
# "cheap" only ever changes which room-rate tier gets picked among hotels
# that already clear this bar, never location quality or star rating.
_HOTEL_MIN_RATING = 4.0

# Ordinal position of each Places (New) priceLevel value, used to measure
# how close a hotel's price is to the tier we're targeting.
_HOTEL_PRICE_RANK = {
    "PRICE_LEVEL_FREE": 0,
    "PRICE_LEVEL_INEXPENSIVE": 1,
    "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
}

# Target rank per budget tier. "luxury" sits between EXPENSIVE(3) and
# VERY_EXPENSIVE(4) rather than locking onto only the priciest result —
# there often isn't a VERY_EXPENSIVE hotel near a given attraction cluster,
# and an EXPENSIVE one right next to the traveler's picks is a better
# recommendation than a VERY_EXPENSIVE one across town.
_HOTEL_TARGET_RANK = {"cheap": 1, "mid": 2, "luxury": 3.5}

# ROOT CAUSE of a real bug (see pick_hotel's docstring): places.
# search_hotels_near's locationBias is only a soft ranking preference in
# Places API (New) Text Search, NOT a hard geographic filter — Google can
# and does return text-relevant results well outside the biased circle.
# Without an explicit distance check, ranking purely by price/rating could
# (and did) pick a hotel over 1000km from the traveler's actual trip. This
# is a hard cutoff enforced BEFORE price/rating ranking, independent of
# whatever radius was used to bias the search itself.
_HOTEL_MAX_DISTANCE_FROM_ANCHOR_KM = 25.0


def _hotel_distance_from_anchor_km(hotel: dict, anchor: tuple[float, float]) -> float | None:
    """Returns the hotel's distance from `anchor` in km, or None if it has no usable coordinates."""
    loc = hotel.get("location") or {}
    lat, lng = loc.get("latitude"), loc.get("longitude")
    if lat is None or lng is None:
        return None
    return _haversine_km(anchor, (lat, lng))


def cluster_center(locations: list[tuple[float | None, float | None]]) -> tuple[float, float] | None:
    """
    Centroid (mean lat, mean lng) of the traveler's selected attractions —
    good enough to anchor a hotel search near "the middle of what they
    picked" without needing a real geographic-clustering algorithm. None
    entries (missing coordinates) are skipped; returns None if nothing
    usable was given.
    """
    valid = [(lat, lng) for lat, lng in locations if lat is not None and lng is not None]
    if not valid:
        return None
    lat = sum(p[0] for p in valid) / len(valid)
    lng = sum(p[1] for p in valid) / len(valid)
    return (lat, lng)


def pick_hotel(hotels: list[dict], budget: str, anchor: tuple[float, float] | None = None) -> dict | None:
    """
    Selects the single best hotel from raw Places hotel-search results
    (places.search_hotels_near) for `budget` ("cheap" | "mid" | "luxury").
    Returns None if no open, 4.0+-rated, geographically-plausible hotel is
    present at all.

    A hotel with no priceLevel at all (common — Places doesn't always set
    it for lodging) is treated as a neutral fit rather than penalized or
    guessed at, and ranked purely by rating in that case — honest
    uncertainty beats a false-precision price match.

    `anchor` should be the same (lat, lng) center passed to
    places.search_hotels_near for this same search — when given, any
    result farther than _HOTEL_MAX_DISTANCE_FROM_ANCHOR_KM is excluded
    BEFORE price/rating ranking (see that constant's comment for why this
    exists — it's a real-bug fix, not defensive paranoia). Pass None only
    when no anchor is available; every current call site always has one.
    """
    eligible = [
        h for h in hotels
        if is_open(h) and (h.get("rating", 0) or 0) >= _HOTEL_MIN_RATING
    ]
    if anchor is not None:
        still_eligible = []
        for h in eligible:
            dist = _hotel_distance_from_anchor_km(h, anchor)
            if dist is None or dist > _HOTEL_MAX_DISTANCE_FROM_ANCHOR_KM:
                name = h.get("displayName", {}).get("text", "?")
                print(f"[TripBuilder] Excluding hotel '{name}' — "
                      f"{'no coordinates' if dist is None else f'{dist:.0f}km from anchor (max {_HOTEL_MAX_DISTANCE_FROM_ANCHOR_KM:.0f}km)'}")
                continue
            still_eligible.append(h)
        eligible = still_eligible
    if not eligible:
        return None

    target_rank = _HOTEL_TARGET_RANK[budget]

    def sort_key(h):
        price_level = h.get("priceLevel", "")
        rank = _HOTEL_PRICE_RANK.get(price_level, target_rank)  # unknown price → neutral fit
        rating = h.get("rating", 0) or 0
        return (abs(rank - target_rank), -rating)

    eligible.sort(key=sort_key)
    return eligible[0]


def hotel_to_recommendation_dict(hotel: dict, city: str) -> dict:
    """
    Converts one raw Places hotel result (places.search_hotels_near),
    already chosen by pick_hotel(), into a TripHotelRecommendation-shaped
    dict — including real Booking.com/Expedia affiliate links built with
    the SAME functions ai_analyzer.py already uses for video-extracted
    hotel stops (_booking_affiliate_url / _expedia_affiliate_url): same
    affiliate accounts, no new link logic written for this feature.
    """
    name = hotel.get("displayName", {}).get("text", "")
    loc = hotel.get("location") or {}
    query = f"{name} {city}".strip()
    return {
        "name": name,
        "description": "",
        "photo_url": photo_url_from_places_photos(hotel.get("photos", [])),
        "rating": hotel.get("rating", 0) or 0,
        "user_rating_count": hotel.get("userRatingCount", 0) or 0,
        "price_level": hotel.get("priceLevel", "") or "",
        "property_type": "",
        "area_label": "",
        "lat": loc.get("latitude"),
        "lng": loc.get("longitude"),
        "booking_url": _booking_affiliate_url(query),
        "expedia_url": _expedia_affiliate_url(query),
    }


# ── Phase C: day-by-day itinerary assembly ──────────────────────────────

def _nearest_neighbor_order(attractions: list[dict]) -> list[dict]:
    """
    Orders `attractions` (each a dict with "lat"/"lng") into a single
    greedy nearest-neighbor path, starting from the westmost point (an
    arbitrary but deterministic choice, so the same selection always
    produces the same order) — a simple, defensible heuristic for
    minimizing backtracking across the whole trip. NOT a full TSP solve;
    Build Your Own Trip doesn't need routing-optimization-grade precision.

    Attractions missing coordinates are appended at the end, in their
    original order — there's nothing to route them by.
    """
    with_coords = [a for a in attractions if a.get("lat") is not None and a.get("lng") is not None]
    without_coords = [a for a in attractions if a.get("lat") is None or a.get("lng") is None]
    if not with_coords:
        return without_coords

    remaining = with_coords[:]
    start = min(remaining, key=lambda a: a["lng"])
    remaining.remove(start)
    path = [start]

    while remaining:
        last = path[-1]
        nxt = min(remaining, key=lambda a: _haversine_km((last["lat"], last["lng"]), (a["lat"], a["lng"])))
        remaining.remove(nxt)
        path.append(nxt)

    return path + without_coords


def group_into_days(attractions: list[dict], num_days: int) -> list[list[dict]]:
    """
    Groups `attractions` into up to `num_days` day-clusters, each already
    ordered by proximity: builds one nearest-neighbor path across ALL
    attractions (_nearest_neighbor_order), then splits it into contiguous
    chunks. A contiguous slice of an already-proximity-ordered path is
    naturally a geographic cluster, and day N+1 picks up near where day N
    ended — this single algorithm satisfies both "group nearby stops
    together" and "order within a day to minimize backtracking" at once.

    Chunk sizes are as even as possible, with any remainder given to the
    earlier days. If there are fewer attractions than `num_days`, trailing
    days get an empty list — assemble_days() drops those rather than
    padding the itinerary with content-free days.
    """
    num_days = max(1, num_days)
    ordered = _nearest_neighbor_order(attractions)
    n = len(ordered)
    base, remainder = divmod(n, num_days)
    days: list[list[dict]] = []
    idx = 0
    for day_i in range(num_days):
        size = base + (1 if day_i < remainder else 0)
        days.append(ordered[idx:idx + size])
        idx += size
    return days


def _candidate_dict_to_stop(candidate: dict) -> dict:
    """
    Converts one selected TripCandidate-shaped dict into a Stop-shaped
    dict. is_specific_name=True: this is a real, confirmed Places result
    the traveler explicitly picked, not an AI guess at a name.
    """
    return {
        "name": candidate.get("name", ""),
        "category": candidate.get("category") or "sight",
        "description": candidate.get("description", ""),
        "photo_url": candidate.get("photo_url", ""),
        "is_specific_name": True,
        "lat": candidate.get("lat"),
        "lng": candidate.get("lng"),
    }


def _hotel_dict_to_stop(hotel: dict) -> dict:
    """Converts the Phase B TripHotelRecommendation-shaped dict into a Stop-shaped dict."""
    return {
        "name": hotel.get("name", ""),
        "category": "hotel",
        "description": hotel.get("description", ""),
        "photo_url": hotel.get("photo_url", ""),
        "is_specific_name": True,
        "property_type": hotel.get("property_type", ""),
        "area_label": hotel.get("area_label", ""),
        "booking_url": hotel.get("booking_url", ""),
        "expedia_url": hotel.get("expedia_url", ""),
        "lat": hotel.get("lat"),
        "lng": hotel.get("lng"),
    }


# Defense-in-depth for the SAME class of bug pick_hotel's anchor filter
# fixes at the source (see _HOTEL_MAX_DISTANCE_FROM_ANCHOR_KM above): any
# Stop — hotel or attraction — could in principle end up with a
# geographically wrong coordinate from a bad Places match. Rather than
# trust every upstream source to always be right, every built itinerary
# gets one final pass: a Stop whose coordinates are wildly far from the
# itinerary's OTHER stops (median-based — robust to a single bad outlier,
# unlike a mean) gets its coordinates dropped and a warning logged, so the
# map view just shows no pin/line for that stop instead of drawing a route
# across half the globe. The stop itself (name/description/card) is never
# removed — only its lat/lng.
_MAX_STOP_DISTANCE_FROM_MEDIAN_KM = 100.0


def _drop_implausible_stop_coordinates(days: list[dict]) -> None:
    """Mutates `days` in place — see the module comment above this function."""
    all_stops = [s for day in days for s in day["stops"]]
    with_coords = [s for s in all_stops if s.get("lat") is not None and s.get("lng") is not None]
    if len(with_coords) < 2:
        return  # nothing to sanity-check against
    median_point = (
        statistics.median(s["lat"] for s in with_coords),
        statistics.median(s["lng"] for s in with_coords),
    )
    for s in with_coords:
        dist = _haversine_km(median_point, (s["lat"], s["lng"]))
        if dist > _MAX_STOP_DISTANCE_FROM_MEDIAN_KM:
            print(f"[TripBuilder] Dropping coordinates for stop '{s.get('name')}' — "
                  f"{dist:.0f}km from the itinerary's other stops (likely a bad Places match)")
            s["lat"] = None
            s["lng"] = None


def assemble_days(attractions: list[dict], num_days: int, hotel: dict | None) -> list[dict]:
    """
    Turns selected TripCandidate-shaped attraction dicts (+ optionally a
    Phase B TripHotelRecommendation-shaped hotel dict) into a list of
    DayPlan-shaped dicts, ready for models.DayPlan(**d).

    JUDGMENT CALL — flagged for review: the hotel Stop always goes first
    in Day 1, matching the same convention ai_analyzer.py's video-
    extraction prompt already uses ("put the hotel in Day 1 — it's the
    base the traveler returns to"). This isn't a new rule invented for
    Phase C, just applied consistently here too.

    JUDGMENT CALL — flagged for review: day labels are plain "Day N" —
    Phase C doesn't run an AI pass to write a themed label (e.g. "Old
    Town & Harbour") the way video extraction does; that would need an
    extra model call this phase's spec didn't ask for.

    JUDGMENT CALL — flagged for review: if `num_days` exceeds how many
    attractions were selected, trailing empty days are dropped — the
    returned itinerary can have fewer days than requested rather than
    padding with content-free days.
    """
    day_groups = group_into_days(attractions, num_days)
    days: list[dict] = []
    for i, group in enumerate(day_groups, start=1):
        stops: list[dict] = []
        if i == 1 and hotel:
            stops.append(_hotel_dict_to_stop(hotel))
        stops.extend(_candidate_dict_to_stop(a) for a in group)
        if not stops:
            continue  # drop empty trailing days
        days.append({"day": i, "label": f"Day {i}", "stops": stops})

    if not days:
        # Edge case: no attractions selected and no hotel found — keep a
        # single empty Day 1 rather than returning an itinerary with zero
        # days at all.
        days = [{"day": 1, "label": "Day 1", "stops": []}]

    _drop_implausible_stop_coordinates(days)
    return days


_CAR_RENTAL_SYSTEM = """You are a travel expert advising whether a traveler visiting a specific destination should rent a car for their trip.

Decide based on REAL-WORLD knowledge of the destination itself — its actual driving conditions, parking availability, traffic, public transit quality, and geographic layout. Do NOT reason from a list of stop names or how spread out they are; reason about the destination as a place, using what you actually know about it.

Examples of the judgment expected:
- New York City: large and spread out, but driving is a nightmare (traffic, parking, one-way grids) and transit/walking covers it well — no car recommended.
- A rural region, a multi-town or island destination, or somewhere with poor public transit — car recommended.
- A dense, walkable European old-town city with good transit — no car recommended, even if the surrounding region is large.
- A destination genuinely requiring travel between towns with no good transit link — car recommended.

Reply with ONLY valid JSON, no markdown fences:
{"car_rental_recommended": true, "car_rental_note": "Short, concrete reason, e.g. 'Public transit doesn't connect these towns well.'"}

car_rental_note must be an empty string if car_rental_recommended is false. Keep the note under 20 words and concrete — not generic filler like "cars are convenient"."""


def recommend_car_rental(destination: str) -> tuple[bool, str, float]:
    """
    Claude Haiku call reasoning with real-world knowledge about whether
    `destination` is a car-rental-worthy trip — deliberately NOT a
    geometric/distance heuristic based on how spread out the selected
    stops are (a big, spread-out city like New York is still a driving
    nightmare; a small but transit-poor island genuinely needs a car).
    Populates the same car_rental_recommended/car_rental_note fields the
    video-extraction pipeline already sets (see ai_analyzer.py's
    SYSTEM_PROMPT) — same contract, different source, same locked logic.

    Returns (recommended, note, cost_usd). Never raises — on any failure
    falls back to (False, "", 0.0), same non-fatal philosophy as the rest
    of this module and quality_check.py.
    """
    try:
        response = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=_CAR_RENTAL_SYSTEM,
            messages=[{"role": "user", "content": f"Destination: {destination}"}],
        )
        usage = getattr(response, "usage", None)
        cost_usd = 0.0
        if usage:
            cost_usd = (
                usage.input_tokens * _HAIKU_INPUT_PER_MTOK
                + usage.output_tokens * _HAIKU_OUTPUT_PER_MTOK
            ) / 1_000_000

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(raw)
        recommended = bool(result.get("car_rental_recommended", False))
        note = str(result.get("car_rental_note") or "").strip() if recommended else ""
        return recommended, note, cost_usd
    except Exception as e:
        print(f"[TripBuilder] Car rental recommendation failed (non-fatal): {type(e).__name__}: {e}")
        return False, "", 0.0
