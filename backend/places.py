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


def _get_place_photo_url(query: str, max_width: int = 800) -> str:
    """Given a search query (e.g. "Cafe 67 Rome"), returns one photo URL or ""."""
    places = _search_places(query, max_results=1)
    if not places:
        print(f"[Places] No places found for '{query}'")
        return ""
    photos = places[0].get("photos", [])
    if not photos:
        print(f"[Places] Place found but no photos for '{query}'")
        return ""
    return _build_photo_url(photos[0].get("name", ""), max_width)


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

    def _best_photo_for_query(query: str) -> str | None:
        """
        Fetches a few candidates for one query and returns the most-liked
        one — a same-query set can range from a striking professional shot
        to someone's blurry vacation snapshot, and Unsplash's default
        ranking doesn't reliably put the best one first for narrow queries.
        """
        try:
            resp = requests.get(
                UNSPLASH_SEARCH_URL,
                params={"query": query, "per_page": 5, "orientation": "landscape"},
                headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
                timeout=6,
            )
            if resp.status_code != 200:
                print(f"[Unsplash] HTTP {resp.status_code} for '{query}': {resp.text[:200]}")
                return None
            results = resp.json().get("results", [])
            if not results:
                return None
            best = max(results, key=lambda r: r.get("likes", 0))
            return best.get("urls", {}).get("regular")
        except Exception as e:
            print(f"[Unsplash] Exception for '{query}': {type(e).__name__}: {e}")
            return None

    photo_urls: list[str] = []
    for query in queries + fallback_queries:
        if len(photo_urls) >= count:
            break
        url = _best_photo_for_query(query)
        if url and url not in photo_urls:
            photo_urls.append(url)

    print(f"[Unsplash] Gallery for '{destination}': {len(photo_urls)} photos")
    return photo_urls


def enrich_itinerary_with_photos(itinerary) -> None:
    """
    Mutates the itinerary in-place:
      - gallery_photo_urls / hero_photo_url: Unsplash (curated destination shots)
      - each stop's photo_url: Google Places (accurate place-specific shots)
    """
    # Destination gallery from Unsplash — first photo doubles as the hero
    gallery = _get_destination_gallery_unsplash(itinerary.destination, count=5)
    itinerary.gallery_photo_urls = gallery
    itinerary.hero_photo_url = gallery[0] if gallery else ""
    print(f"[Photos] Hero: {itinerary.destination} → {bool(itinerary.hero_photo_url)}")

    if not PLACES_API_KEY:
        print("[Places] No API key — skipping stop photo enrichment")
        return

    # City/region name only (e.g. "Bali" from "Bali, Indonesia") — appended
    # to every stop's search query below. Without it, generic or ambiguous
    # stop names (e.g. "Diamond Beach", which also exists in Iceland and
    # Australia) can match a place on the wrong side of the world, pulling
    # back a photo — and a Maps link — that has nothing to do with the trip.
    city = itinerary.destination.split(",")[0].strip()

    # Photo for each stop — still Google Places (accurate for named locations)
    for day in itinerary.days:
        for stop in day.stops:
            query = f"{stop.name}, {city}" if city else stop.name
            stop.photo_url = _get_place_photo_url(query)
            print(f"[Places] Stop: {query} → {bool(stop.photo_url)}")
