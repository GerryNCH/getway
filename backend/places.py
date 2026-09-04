"""
places.py — Google Places API integration (stops) + Unsplash (destination photos).

Places API (New) is used for specific named stops (hotels, restaurants,
landmarks) where it returns accurate, place-specific photos.

Unsplash is used for the destination hero/gallery, because Places API
photos for a bare city/region query are often low-quality or blurry
(user-submitted snapshots), whereas Unsplash returns curated, high-res
travel photography — exactly what a hero banner needs.

Free tier: Places $200/month credit (~5000 lookups). Unsplash: 50 req/hour
on the free Demo tier — plenty for this use case.
"""

import os
import re
import time
import requests

import database
from models import UnsplashAttribution
from ai_analyzer import _booking_affiliate_url, _expedia_affiliate_url

PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "")

SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
UNSPLASH_SEARCH_URL = "https://api.unsplash.com/search/photos"


def _build_photo_url(photo_name: str, max_width: int = 1600) -> str:
    return (
        f"https://places.googleapis.com/v1/{photo_name}/media"
        f"?maxWidthPx={max_width}&key={PLACES_API_KEY}&skipHttpRedirect=false"
    )


def photo_url_from_places_photos(photos: list[dict], max_width: int = 800) -> str:
    """
    Picks the best available photo (widest landscape shot, falling back to
    widest overall) from a raw Places `photos` array and returns its URL,
    or "" if the array is empty. Shared by trip_builder.py's candidate and
    hotel result conversion — both need this exact same "pick one good
    photo" logic.
    """
    if not photos:
        return ""
    landscape = [p for p in photos if p.get("widthPx", 0) > p.get("heightPx", 0)]
    best = max(landscape or photos, key=lambda p: p.get("widthPx", 0))
    return _build_photo_url(best.get("name", ""), max_width)


def _search_places(query: str, max_results: int = 1, _retries: int = 2) -> list[dict]:
    """
    Runs a Places Text Search and returns the raw places list (with photos
    field). Retries once on transient failures (timeout, connection error,
    429, 5xx) before giving up — a single retry recovers most of these, and
    the alternative is silently losing that stop's coordinates/photo
    forever, since this same call is what populates Stop.lat/lng at
    generation time. Does NOT retry genuine "not found" cases (a 200
    response with an empty/non-matching result) — retrying an unresolvable
    generic name just wastes calls.
    """
    if not PLACES_API_KEY:
        return []
    for attempt in range(_retries):
        is_last_attempt = attempt == _retries - 1
        try:
            resp = requests.post(
                SEARCH_URL,
                json={"textQuery": query, "maxResultCount": max_results},
                headers={
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": PLACES_API_KEY,
                    "X-Goog-FieldMask": "places.photos,places.displayName,places.location",
                },
                timeout=6,
            )
            if resp.status_code == 200:
                return resp.json().get("places", [])
            if resp.status_code == 429 or resp.status_code >= 500:
                # Transient (rate limit / server-side) — worth a retry.
                if not is_last_attempt:
                    time.sleep(0.6)
                    continue
            print(f"[Places] HTTP {resp.status_code} for '{query}': {resp.text[:300]}")
            return []
        except requests.exceptions.RequestException as e:
            if not is_last_attempt:
                time.sleep(0.6)
                continue
            print(f"[Places] Exception searching '{query}' after {_retries} attempts: {type(e).__name__}: {e}")
            return []
    return []


# Field mask for the BROAD "Build Your Own Trip" candidate search
# (search_attractions_broad, search_activities_by_type) — separate from
# _search_places' minimal mask above (photos/displayName/location only,
# tuned for confirming ONE named place already known to exist). This one
# needs the extra signal fields Build Your Own Trip's budget classification
# and AI curation pass depend on (see trip_builder.py): priceLevel
# (paid-tier classification), rating/userRatingCount (famous-place override
# + curation input), types (free-by-type classification + junk filtering),
# businessStatus (drop permanently/temporarily closed places), id (lets
# trip_builder.dedupe_places identify the SAME real place returned by two
# different searches — e.g. the attraction search and an activity-type
# search both surfacing the same famous landmark — reliably, instead of
# only via a fuzzy displayName match).
_ATTRACTION_SEARCH_FIELD_MASK = (
    "places.id,places.displayName,places.location,places.photos,places.priceLevel,"
    "places.rating,places.userRatingCount,places.types,places.businessStatus"
)


def _text_search(query: str, field_mask: str, max_results: int = 20, _retries: int = 2) -> list[dict]:
    """
    Shared Places API (New) Text Search POST + retry/error-handling logic —
    used by search_attractions_broad and search_activities_by_type, which
    are identical except for the query string. NOT used by
    search_hotels_near, which needs an extra locationBias body field.
    """
    if not PLACES_API_KEY:
        return []
    for attempt in range(_retries):
        is_last_attempt = attempt == _retries - 1
        try:
            resp = requests.post(
                SEARCH_URL,
                json={"textQuery": query, "maxResultCount": max_results},
                headers={
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": PLACES_API_KEY,
                    "X-Goog-FieldMask": field_mask,
                },
                timeout=8,
            )
            if resp.status_code == 200:
                return resp.json().get("places", [])
            if resp.status_code == 429 or resp.status_code >= 500:
                if not is_last_attempt:
                    time.sleep(0.6)
                    continue
            print(f"[Places] Text search HTTP {resp.status_code} for '{query}': {resp.text[:300]}")
            return []
        except requests.exceptions.RequestException as e:
            if not is_last_attempt:
                time.sleep(0.6)
                continue
            print(f"[Places] Text search exception for '{query}' after {_retries} attempts: {type(e).__name__}: {e}")
            return []
    return []


def search_attractions_broad(city: str, max_results: int = 20, _retries: int = 2) -> list[dict]:
    """
    Broad Google Places Text Search for "Build Your Own Trip" candidate
    attractions in `city` (e.g. "top attractions and things to do in
    Lisbon, Portugal"). Returns the raw places list with the expanded
    field mask above — unclassified and uncurated; see trip_builder.py for
    budget classification and the AI curation pass.

    max_results=20 is the Places API (New) Text Search hard cap for a
    single call (no pagination here) — plenty for one destination's
    candidate pool once combined with the AI curation pass.
    """
    query = f"top attractions and things to do in {city}"
    return _text_search(query, _ATTRACTION_SEARCH_FIELD_MASK, max_results, _retries)


# Activity-type slug -> Places Text Search query phrase, for the Build Your
# Own Trip "what kind of activities?" wizard step. Slugs are what the
# frontend/backend pass around; the phrases are only ever used to build a
# search query here.
_ACTIVITY_TYPE_QUERY_TERMS = {
    "nightlife": "nightlife tours, bars and clubs",
    "nature": "guided hiking tours and outdoor excursions",
    "history": "guided historical walking tours",
    "beach": "boat tours, sailing and water sports",
    "food": "food tours, cooking classes and tastings",
    "art": "art workshops and painting classes",
    "adventure": "adventure tours and adrenaline activities",
    "family": "family-friendly tours and activities",
}


def search_activities_by_type(city: str, activity_type: str, max_results: int = 15, _retries: int = 2) -> list[dict]:
    """
    Text Search for one specific activity-type slug (see
    _ACTIVITY_TYPE_QUERY_TERMS) in `city` — e.g. "nightlife, bars and clubs
    in Lisbon, Portugal". Returns [] for an unrecognized slug (main.py
    already normalizes/drops unknown slugs before calling this — this is
    just a second line of defense) rather than raising. Same field mask
    and retry/error handling as search_attractions_broad; results are
    unclassified/uncurated — main.py merges them with the plain attraction
    search results before budget filtering and AI curation.
    """
    phrase = _ACTIVITY_TYPE_QUERY_TERMS.get(activity_type)
    if not phrase:
        return []
    query = f"{phrase} in {city}"
    return _text_search(query, _ATTRACTION_SEARCH_FIELD_MASK, max_results, _retries)


def search_places_freetext(city: str, query: str, max_results: int = 10, _retries: int = 2) -> list[dict]:
    """
    Free-text Places Text Search for a specific traveler-typed query (e.g.
    "diving") in `city` — Build Your Own Trip's manual search-and-add
    feature, for something the curated attraction/activity-type candidate
    lists didn't surface. Same field mask and retry/error handling as
    search_attractions_broad; results are unclassified/uncurated — main.py
    runs them through the same AI curation pass as everything else (no
    budget filter, though — the traveler explicitly asked for this).
    """
    q = f"{query} in {city}"
    return _text_search(q, _ATTRACTION_SEARCH_FIELD_MASK, max_results, _retries)


# Field mask for the "Build Your Own Trip" hotel search (search_hotels_near)
# — same signal fields as the attraction search above (priceLevel, rating,
# userRatingCount, businessStatus) plus formattedAddress, which the plain
# attraction search doesn't need but a hotel recommendation benefits from
# for a rough area description.
_HOTEL_SEARCH_FIELD_MASK = (
    "places.displayName,places.location,places.photos,places.priceLevel,"
    "places.rating,places.userRatingCount,places.businessStatus,places.formattedAddress"
)


def search_hotels_near(lat: float, lng: float, city: str, radius_meters: int = 4000,
                        max_results: int = 20, _retries: int = 2) -> list[dict]:
    """
    Geographically-anchored hotel search: a "hotels in {city}" Text Search
    biased toward (lat, lng) via Places API (New)'s locationBias, so
    results cluster around wherever the traveler's selected attractions
    actually are — not wherever Google's own city-center bias happens to
    land a plain "hotels in {city}" query. This is what makes the LOCKED
    LOGIC guarantee possible ("cheap" budget never means a worse location):
    trip_builder.pick_hotel() filters/ranks these results, but it can only
    pick from what's actually near the traveler's chosen stops.

    locationBias is a soft preference, not a hard filter — Google can still
    return something outside `radius_meters` if nothing better exists.
    Returns the raw places list with the field mask above; unclassified —
    see trip_builder.pick_hotel() for the 4.0+ rating floor and budget fit.
    """
    if not PLACES_API_KEY:
        return []
    query = f"hotels in {city}"
    body = {
        "textQuery": query,
        "maxResultCount": max_results,
        "locationBias": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": float(radius_meters),
            }
        },
    }
    for attempt in range(_retries):
        is_last_attempt = attempt == _retries - 1
        try:
            resp = requests.post(
                SEARCH_URL,
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": PLACES_API_KEY,
                    "X-Goog-FieldMask": _HOTEL_SEARCH_FIELD_MASK,
                },
                timeout=8,
            )
            if resp.status_code == 200:
                return resp.json().get("places", [])
            if resp.status_code == 429 or resp.status_code >= 500:
                if not is_last_attempt:
                    time.sleep(0.6)
                    continue
            print(f"[Places] Hotel search HTTP {resp.status_code} for '{city}' near ({lat}, {lng}): {resp.text[:300]}")
            return []
        except requests.exceptions.RequestException as e:
            if not is_last_attempt:
                time.sleep(0.6)
                continue
            print(f"[Places] Hotel search exception for '{city}' after {_retries} attempts: {type(e).__name__}: {e}")
            return []
    return []


def _names_plausibly_match(query_name: str, candidate_name: str) -> bool:
    """
    Loose check that a Places search result is actually the place we asked
    for, not just whatever Google's text search happened to rank first.
    Word-overlap based (not exact match) since displayName often differs
    slightly in wording/punctuation from how the AI wrote the stop name.
    """
    strip = lambda s: re.sub(r"[^\w\s]", " ", s.lower())
    stopwords = {"the", "a", "an", "of", "at", "in", "and", "cafe", "restaurant"}
    q_words = {w for w in strip(query_name).split() if len(w) > 2 and w not in stopwords}
    c_words = {w for w in strip(candidate_name).split() if len(w) > 2 and w not in stopwords}
    if not q_words:
        return True
    overlap = q_words & c_words
    return len(overlap) / len(q_words) >= 0.4


def _location_from_place(place: dict) -> tuple[float, float] | None:
    """
    Extracts (lat, lng) from a Places API (New) result's `location` field
    (only present now that the field mask includes "places.location").
    Returns None when absent rather than (0, 0), which would otherwise
    silently plot a stop in the Gulf of Guinea on the route Map view.
    """
    loc = place.get("location") or {}
    lat, lng = loc.get("latitude"), loc.get("longitude")
    if lat is None or lng is None:
        return None
    return (lat, lng)


def _get_place_photo_and_location(query: str, max_width: int = 800) -> tuple[str, tuple[float, float] | None]:
    """
    Same name-matching logic as _get_place_photo_url, but also returns the
    matched place's (lat, lng) — used to populate Stop.lat/Stop.lng for the
    route Map view (index.html). Costs no extra API call: the coordinates
    ride along on the same Text Search response we already fetch for the
    photo. Returns ("", None) when nothing confidently matches; returns
    ("", location) when a confident name match has coordinates but no
    usable photo — a stop can still get a map pin without a photo.
    """
    places = _search_places(query, max_results=3)
    if not places:
        print(f"[Places] No places found for '{query}'")
        return "", None

    query_core = query.split(",")[0]  # drop the appended city for the name comparison
    for place in places:
        name = place.get("displayName", {}).get("text", "")
        if not _names_plausibly_match(query_core, name):
            continue
        location = _location_from_place(place)
        photos = place.get("photos", [])
        if not photos:
            return "", location
        candidates = photos[:5]
        landscape = [p for p in candidates if p.get("widthPx", 0) > p.get("heightPx", 0)]
        best = max(landscape or candidates, key=lambda p: p.get("widthPx", 0))
        return _build_photo_url(best.get("name", ""), max_width), location

    print(f"[Places] No confident name match for '{query}' — skipping photo rather than risk a wrong one")
    return "", None


def _get_place_photo_url(query: str, max_width: int = 800) -> str:
    """
    Given a search query (e.g. "Cafe 67, Rome"), returns one photo URL or
    "". Thin wrapper around _get_place_photo_and_location for call sites
    that only need the photo — kept so nothing else has to change.
    """
    url, _ = _get_place_photo_and_location(query, max_width)
    return url


def _unsplash_candidates(query: str, per_page: int = 6) -> list[dict]:
    """
    Fetches raw Unsplash search results (id, urls, likes, user, links) for
    one query. Excludes Unsplash+ ("plus") photos — those are a separate
    paid license tier and get served with a visible tiled watermark unless
    the requesting app has an Unsplash+ subscription, which this app
    doesn't. Regular free-tier Unsplash photos have no such restriction.
    """
    if not UNSPLASH_ACCESS_KEY:
        return []
    try:
        resp = requests.get(
            UNSPLASH_SEARCH_URL,
            params={"query": query, "per_page": per_page, "orientation": "landscape"},
            headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
            timeout=6,
        )
        # Unsplash returns these on every response (even successful ones) —
        # logging them means a rate-limit problem shows up immediately in
        # Railway logs instead of being guessed at after the fact. Demo-tier
        # apps get 50/hour; Production-tier gets 5000/hour.
        limit = resp.headers.get("X-Ratelimit-Limit")
        remaining = resp.headers.get("X-Ratelimit-Remaining")
        if limit and remaining:
            print(f"[Unsplash] Rate limit: {remaining}/{limit} remaining this hour")
        if resp.status_code != 200:
            print(f"[Unsplash] HTTP {resp.status_code} for '{query}': {resp.text[:200]}")
            return []
        results = resp.json().get("results", [])
        return [r for r in results if not r.get("plus")]
    except Exception as e:
        print(f"[Unsplash] Exception for '{query}': {type(e).__name__}: {e}")
        return []


def _attribution_from_candidate(c: dict) -> UnsplashAttribution:
    """
    Extracts the fields Unsplash's API guidelines require us to display
    whenever a photo is shown: the photographer's name + profile link, and
    a link to the photo's own Unsplash page.
    """
    user = c.get("user") or {}
    return UnsplashAttribution(
        photographer_name=user.get("name", ""),
        photographer_url=(user.get("links") or {}).get("html", ""),
        unsplash_url=(c.get("links") or {}).get("html", ""),
    )


def _trigger_unsplash_download(c: dict) -> None:
    """
    Fires Unsplash's required "download" tracking event for a photo that's
    actually being used (not just browsed in search results) — part of
    their API guidelines for Production access. Best-effort: this should
    never block or fail the actual response to the user.
    """
    download_location = (c.get("links") or {}).get("download_location")
    if not download_location or not UNSPLASH_ACCESS_KEY:
        return
    try:
        requests.get(
            download_location,
            headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
            timeout=4,
        )
    except Exception as e:
        print(f"[Unsplash] Download-tracking ping failed (non-fatal): {type(e).__name__}: {e}")


def _get_destination_gallery_unsplash(destination: str, count: int = 5) -> list[dict]:
    """
    Returns up to `count` curated, high-resolution travel photo entries for
    a destination from Unsplash, each as {"url": ..., "likes": ...}. Uses
    several distinct queries (rather than one broad query) so the gallery
    shows varied shots instead of several near-duplicate frames from the
    same photo session. Falls back to an empty list if no key is
    configured or all requests fail.
    """
    if not UNSPLASH_ACCESS_KEY:
        print("[Unsplash] No API key — skipping destination gallery")
        return []

    # Multi-city destinations (e.g. "Cairo & Luxor", "Rome and Florence")
    # need to be split down to a single clean city name — querying Unsplash
    # with the full compound string returns few or no results, which starved
    # the gallery down to 0-1 photos and hid the thumbnail strip entirely.
    city = re.split(r"\s*(?:,|&|\band\b)\s*", destination, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    # These target generically striking travel photography rather than
    # city-specific shots — "night skyline" or "waterfront" return nothing
    # useful for an island/nature destination (e.g. Bali), which is what
    # starved the gallery down to 2-3 photos instead of 5. "Aerial",
    # "sunset", "scenic", and "beautiful" are terms photographers tag
    # constantly across every destination type, so they reliably surface
    # a full gallery of appealing shots for cities, islands, and nature
    # destinations alike.
    queries = [
        f"{city} aerial view",
        f"{city} scenic",
        f"{city} sunset",
        f"{city} landmark",
        f"{city} beautiful",
    ]
    # Used only to top up the gallery if the specific queries above didn't
    # collectively return `count` photos (e.g. an obscure destination) —
    # broad enough to almost always return something, still tied to the
    # destination rather than falling back to something generic like
    # "travel", which could return a photo of an unrelated place.
    fallback_queries = [f"{city} travel", f"{city} view", city]

    # Overlapping queries (e.g. "Cairo aerial view" and "Cairo landmark")
    # very often surface the exact same handful of iconic photos as each
    # other's top result — deduping on URL alone let 3 near-identical
    # skyline shots into the same gallery. Tracking used photo IDs *across
    # every query* and falling through to the next-best candidate within
    # a query (instead of giving up on that query) fixes it properly.
    #
    # IMPORTANT: always sample at least 2 distinct queries and pool their
    # candidates together before picking the best `count`, even when
    # count=1. Stopping the loop the moment len(photos) >= count meant
    # count=1 (set to conserve Unsplash quota) searched ONLY the first
    # query term ("aerial view") and kept whatever came back — no chance
    # to compare against "scenic", "landmark", etc. That's what was
    # producing consistently mediocre hero photos, not a lack of good
    # photos on Unsplash. Comparing across a small pool first, then
    # keeping only the top `count`, costs one extra API call but fixes
    # the actual quality regression.
    sample_query_count = max(2, count)
    used_ids: set[str] = set()
    candidate_pool: list[dict] = []
    queries_to_try = (queries + fallback_queries)[:sample_query_count]
    for query in queries_to_try:
        for c in _unsplash_candidates(query):
            cid = c.get("id")
            url = c.get("urls", {}).get("regular")
            if not url or cid in used_ids:
                continue
            used_ids.add(cid)
            candidate_pool.append({
                "url": url,
                "likes": c.get("likes", 0),
                "attribution": _attribution_from_candidate(c),
                "_raw": c,
            })

    candidate_pool.sort(key=lambda p: p["likes"], reverse=True)
    photos = candidate_pool[:count]
    for p in photos:
        _trigger_unsplash_download(p.pop("_raw"))

    # If the small sample somehow didn't fill `count` (rare — an obscure
    # destination with thin Unsplash coverage), top up from the remaining
    # fallback queries before giving up.
    if len(photos) < count:
        for query in (queries + fallback_queries)[sample_query_count:]:
            if len(photos) >= count:
                break
            for c in sorted(_unsplash_candidates(query), key=lambda r: r.get("likes", 0), reverse=True):
                cid = c.get("id")
                url = c.get("urls", {}).get("regular")
                if not url or cid in used_ids:
                    continue
                used_ids.add(cid)
                _trigger_unsplash_download(c)
                photos.append({"url": url, "likes": c.get("likes", 0), "attribution": _attribution_from_candidate(c)})
                break

    print(f"[Unsplash] Gallery for '{destination}': {len(photos)} photos (sampled {len(queries_to_try)} queries, {len(candidate_pool)} candidates)")
    return photos


# Google Places is now tried first for every specific-name stop regardless
# of category — it has real photos of the actual entity (crowd-sourced from
# Google Maps), whereas Unsplash is keyword-matched stock photography that
# can return something merely thematically similar rather than the actual
# restaurant/beach/landmark. Unsplash-by-name is only the fallback when
# Places has no listing/photo for that specific place.

# Rough Unsplash query term per category — used as a fallback photo when a
# stop's name isn't a real, searchable place (see enrich_itinerary_with_photos).
_CATEGORY_PHOTO_TERMS = {
    "hotel": "hotel room interior",
    "food": "cafe restaurant food",
    "sight": "landmark",
    "activity": "adventure activity",
    "beach": "tropical beach",
    "village": "village street",
}


def enrich_itinerary_with_photos(itinerary) -> None:
    """
    Mutates the itinerary in-place:
      - gallery_photo_urls / hero_photo_url: Unsplash (curated destination shots)
      - each stop's photo_url: Google Places first for ANY specific-name
        stop (real photo of the actual entity, any category); Unsplash
        search-by-name as fallback when Places has no listing/photo for
        that specific place; a category-matched Unsplash photo as the
        final fallback when the AI couldn't confirm a specific name at all
        (`is_specific_name=False`).
    """
    # Destination gallery from Unsplash — the *best-liked* photo doubles as
    # the hero, not just whichever of the 5 queries happened to run first
    # (that previously meant "aerial view" always won the hero slot even
    # when "sunset" or "landmark" returned a much more striking photo).
    gallery = _get_destination_gallery_unsplash(itinerary.destination, count=1)
    itinerary.gallery_photo_urls = [p["url"] for p in gallery]
    itinerary.gallery_attributions = [p["attribution"] for p in gallery]
    hero = max(gallery, key=lambda p: p["likes"]) if gallery else None
    itinerary.hero_photo_url = hero["url"] if hero else ""
    itinerary.hero_attribution = hero["attribution"] if hero else None
    print(f"[Photos] Hero: {itinerary.destination} → {bool(itinerary.hero_photo_url)}")

    # Single primary city (e.g. "Cairo" from "Cairo & Luxor") — same split
    # used for the gallery above. Deliberately NOT the full multi-city
    # string: appending "Cairo & Luxor" to every stop's search query would
    # bias a Luxor stop's photo/Maps search toward Cairo just because it
    # shares the trip. Imperfect for multi-city trips (a Luxor-only stop
    # with no city in its own name still gets "Cairo" appended), but the
    # AI usually already writes the correct city into stop.name itself
    # (e.g. "Egyptian Museum, Cairo") — the fallback below only kicks in
    # when it didn't.
    city = re.split(r"\s*(?:,|&|\band\b)\s*", itinerary.destination, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    used_ids: set[str] = set()  # avoid repeating a photo across stop cards

    def _best_fresh_unsplash(query: str) -> tuple[str, dict | None]:
        """Returns (url, attribution) — attribution is None if nothing found."""
        for c in sorted(_unsplash_candidates(query, per_page=10), key=lambda r: r.get("likes", 0), reverse=True):
            cid, url = c.get("id"), c.get("urls", {}).get("regular")
            if url and cid not in used_ids:
                used_ids.add(cid)
                _trigger_unsplash_download(c)
                return url, _attribution_from_candidate(c)
        return "", None

    for day in itinerary.days:
        for stop in day.stops:
            is_specific = getattr(stop, "is_specific_name", True)
            name_query = stop.name if city.lower() in stop.name.lower() else f"{stop.name}, {city}"

            # Reuse a photo/address already saved from a PREVIOUS route for
            # this exact city+stop name (e.g. the Colosseum showing up
            # again in a second Rome route) — skips the Places/Unsplash
            # lookup entirely, which both saves the API call and means
            # Gerry's earlier manual fix for this stop carries over
            # automatically instead of needing to be redone every time.
            cached = database.get_cached_stop(city, stop.name)
            if cached:
                stop.photo_url = cached["photo_url"]
                if cached.get("maps_url_override"):
                    stop.maps_url_override = cached["maps_url_override"]
                # Only present once database.py's get_cached_stop is updated
                # to also select/return lat/lng — safe no-op until then.
                if cached.get("lat") is not None and cached.get("lng") is not None:
                    stop.lat = cached["lat"]
                    stop.lng = cached["lng"]
                print(f"[Cache] Reused saved photo for '{stop.name}' in {city}")
                continue

            # Unconfirmed HOTEL stops get one extra chance before falling
            # back to generic photos: the AI's own description (e.g.
            # "Beachfront resort hotel, Cala Sant Vicenç area") is often
            # specific enough for Places to surface an actual, real,
            # bookable hotel that matches the style/area — even though it's
            # not confirmed as the exact one shown in the video. When that
            # happens, upgrade the stop to that real hotel: real name, real
            # photo, and real booking links (previously there was no way to
            # "Book this hotel" at all here, since the AI's own description
            # isn't a real bookable entity — only a generic search was
            # possible). is_specific_name stays False so the frontend still
            # shows the "similar match, not confirmed" disclaimer — this is
            # a real place, just not confirmed as THE place from the clip.
            if stop.category == "hotel" and not is_specific and PLACES_API_KEY:
                candidates = _search_places(name_query, max_results=1)
                if candidates:
                    real_name = candidates[0].get("displayName", {}).get("text", "").strip()
                    photos = candidates[0].get("photos", [])
                    if real_name and photos:
                        print(f"[Places] Upgraded unconfirmed hotel '{stop.name}' → real match '{real_name}'")
                        stop.name = real_name
                        stop.similar_hotel_is_real = True
                        stop.photo_url, location = _get_place_photo_and_location(f"{real_name}, {city}")
                        if location:
                            stop.lat, stop.lng = location
                        booking_query = f"{real_name}, {city}"
                        stop.booking_url = _booking_affiliate_url(booking_query)
                        stop.expedia_url = _expedia_affiliate_url(booking_query)

            if is_specific and PLACES_API_KEY:
                # Don't double up the city if the AI already wrote it into
                # the name (e.g. "Nile Corniche, Cairo") — searching
                # "Nile Corniche, Cairo, Cairo & Luxor" is redundant and
                # doesn't help Places match the right place.
                stop.photo_url, location = _get_place_photo_and_location(name_query)  # Places photo — no Unsplash attribution needed
                if location:
                    stop.lat, stop.lng = location
                print(f"[Places] Stop: {name_query} → photo={bool(stop.photo_url)} coords={bool(location)}")

            if not stop.photo_url and is_specific:
                # Places had no listing/photo for this specific place (or no
                # API key configured) — fall back to an Unsplash search BY
                # NAME. This is a weaker match than Places (Unsplash is
                # keyword-matched stock photography, not a business photo
                # directory — it can return something merely thematically
                # similar rather than the actual place), so it only kicks in
                # when Places genuinely has nothing.
                stop.photo_url, stop.photo_attribution = _best_fresh_unsplash(name_query)
                print(f"[Photos] Unsplash (named) fallback for '{stop.name}' → {bool(stop.photo_url)}")

            if stop.photo_url:
                continue

            # Either the AI flagged this as a generic/invented name, there's
            # no API key configured, or the attempt above found nothing.
            # Even when a stop isn't a specific bookable place, its own
            # name is usually a much better photo search term than a
            # generic category label — "ATV / Quad Bike Rental, Santorini"
            # should find ATV photos, not just generic island scenery.
            # Only fall back to the category term if that search comes up
            # empty (e.g. the name is too much of a sentence to match).
            stop.photo_is_fallback = True  # reaching this point always means
                                             # no confident name-match photo —
                                             # flagged for the admin panel.
            stop.photo_url, stop.photo_attribution = _best_fresh_unsplash(name_query)
            if not stop.photo_url:
                term = _CATEGORY_PHOTO_TERMS.get(stop.category, "travel")
                stop.photo_url, stop.photo_attribution = _best_fresh_unsplash(f"{city} {term}".strip())
            print(f"[Photos] Fallback for '{stop.name}' → {bool(stop.photo_url)}")
