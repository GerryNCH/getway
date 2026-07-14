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
import requests

PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "")

SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
UNSPLASH_SEARCH_URL = "https://api.unsplash.com/search/photos"


def _build_photo_url(photo_name: str, max_width: int = 1600) -> str:
    return (
        f"https://places.googleapis.com/v1/{photo_name}/media"
        f"?maxWidthPx={max_width}&key={PLACES_API_KEY}&skipHttpRedirect=false"
    )


def _search_places(query: str, max_results: int = 1) -> list[dict]:
    """Runs a Places Text Search and returns the raw places list (with photos field)."""
    if not PLACES_API_KEY:
        return []
    try:
        resp = requests.post(
            SEARCH_URL,
            json={"textQuery": query, "maxResultCount": max_results},
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": PLACES_API_KEY,
                "X-Goog-FieldMask": "places.photos,places.displayName",
            },
            timeout=6,
        )
        if resp.status_code != 200:
            print(f"[Places] HTTP {resp.status_code} for '{query}': {resp.text[:300]}")
            return []
        return resp.json().get("places", [])
    except Exception as e:
        print(f"[Places] Exception searching '{query}': {type(e).__name__}: {e}")
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


def _get_place_photo_url(query: str, max_width: int = 800) -> str:
    """
    Given a search query (e.g. "Cafe 67, Rome"), returns one photo URL or
    "". Requests a few candidates and picks the first one whose name
    plausibly matches what we searched for — Google's Text Search can
    return an unrelated nearby business as the top hit for a loosely-worded
    or partial-match query, which previously produced photos with nothing
    to do with the actual stop (e.g. a convenience store for "Nile
    Corniche"). If nothing matches confidently, returns "" rather than a
    misleading photo.
    """
    places = _search_places(query, max_results=3)
    if not places:
        print(f"[Places] No places found for '{query}'")
        return ""

    query_core = query.split(",")[0]  # drop the appended city for the name comparison
    for place in places:
        name = place.get("displayName", {}).get("text", "")
        if not _names_plausibly_match(query_core, name):
            continue
        photos = place.get("photos", [])
        if photos:
            return _build_photo_url(photos[0].get("name", ""), max_width)

    print(f"[Places] No confident name match for '{query}' — skipping photo rather than risk a wrong one")
    return ""


def _unsplash_candidates(query: str, per_page: int = 6) -> list[dict]:
    """Fetches raw Unsplash search results (id, urls, likes) for one query."""
    if not UNSPLASH_ACCESS_KEY:
        return []
    try:
        resp = requests.get(
            UNSPLASH_SEARCH_URL,
            params={"query": query, "per_page": per_page, "orientation": "landscape"},
            headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
            timeout=6,
        )
        if resp.status_code != 200:
            print(f"[Unsplash] HTTP {resp.status_code} for '{query}': {resp.text[:200]}")
            return []
        return resp.json().get("results", [])
    except Exception as e:
        print(f"[Unsplash] Exception for '{query}': {type(e).__name__}: {e}")
        return []


def _get_destination_gallery_unsplash(destination: str, count: int = 5) -> list[str]:
    """
    Returns up to `count` curated, high-resolution travel photo URLs for a
    destination from Unsplash. Uses several distinct queries (rather than
    one broad query) so the gallery shows varied shots of the destination
    instead of several near-duplicate frames from the same photo session.
    Falls back to an empty list if no key is configured or all requests fail.
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
    used_ids: set[str] = set()
    photo_urls: list[str] = []
    for query in queries + fallback_queries:
        if len(photo_urls) >= count:
            break
        candidates = sorted(_unsplash_candidates(query), key=lambda r: r.get("likes", 0), reverse=True)
        for c in candidates:
            cid = c.get("id")
            url = c.get("urls", {}).get("regular")
            if not url or cid in used_ids:
                continue
            used_ids.add(cid)
            photo_urls.append(url)
            break  # took the best fresh candidate from this query, move on

    print(f"[Unsplash] Gallery for '{destination}': {len(photo_urls)} photos")
    return photo_urls


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
      - each stop's photo_url: Google Places for real, specifically-named
        stops; a category-matched Unsplash photo for stops the AI couldn't
        confirm a specific name for (`is_specific_name=False`) — searching
        Places with an invented description (e.g. "Café with Nile view,
        Zamalek") just returns nothing, so there's no point trying.
    """
    # Destination gallery from Unsplash — first photo doubles as the hero
    gallery = _get_destination_gallery_unsplash(itinerary.destination, count=5)
    itinerary.gallery_photo_urls = gallery
    itinerary.hero_photo_url = gallery[0] if gallery else ""
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
    used_gallery_ids = set()  # avoid repeating a gallery photo on a stop card

    for day in itinerary.days:
        for stop in day.stops:
            is_specific = getattr(stop, "is_specific_name", True)

            if is_specific and PLACES_API_KEY:
                # Don't double up the city if the AI already wrote it into
                # the name (e.g. "Nile Corniche, Cairo") — searching
                # "Nile Corniche, Cairo, Cairo & Luxor" is redundant and
                # doesn't help Places match the right place.
                query = stop.name if city.lower() in stop.name.lower() else f"{stop.name}, {city}"
                stop.photo_url = _get_place_photo_url(query)
                print(f"[Places] Stop: {query} → {bool(stop.photo_url)}")
                if stop.photo_url:
                    continue

            # Either the AI flagged this as a generic/invented name, there's
            # no Places key configured, or Places found nothing confident —
            # fall back to a representative (not misleading) category photo.
            term = _CATEGORY_PHOTO_TERMS.get(stop.category, "travel")
            fallback_query = f"{city} {term}".strip()
            candidates = sorted(_unsplash_candidates(fallback_query, per_page=6), key=lambda r: r.get("likes", 0), reverse=True)
            for c in candidates:
                cid = c.get("id")
                url = c.get("urls", {}).get("regular")
                if url and cid not in used_gallery_ids:
                    used_gallery_ids.add(cid)
                    stop.photo_url = url
                    break
            print(f"[Photos] Fallback for '{stop.name}' ({fallback_query}) → {bool(stop.photo_url)}")
