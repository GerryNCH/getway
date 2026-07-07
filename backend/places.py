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

    city = destination.split(",")[0].strip()
    queries = [
        f"{city} landmark",
        f"{city} aerial view",
        f"{city} old town",
        f"{city} skyline",
        f"{city} street",
    ]

    photo_urls: list[str] = []
    for query in queries:
        if len(photo_urls) >= count:
            break
        try:
            resp = requests.get(
                UNSPLASH_SEARCH_URL,
                params={"query": query, "per_page": 1, "orientation": "landscape"},
                headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
                timeout=6,
            )
            if resp.status_code != 200:
                print(f"[Unsplash] HTTP {resp.status_code} for '{query}': {resp.text[:200]}")
                continue
            results = resp.json().get("results", [])
            if results and "urls" in results[0]:
                url = results[0]["urls"]["regular"]
                if url not in photo_urls:
                    photo_urls.append(url)
        except Exception as e:
            print(f"[Unsplash] Exception for '{query}': {type(e).__name__}: {e}")
            continue

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

    # Photo for each stop — still Google Places (accurate for named locations)
    for day in itinerary.days:
        for stop in day.stops:
            stop.photo_url = _get_place_photo_url(stop.name)
            print(f"[Places] Stop: {stop.name} → {bool(stop.photo_url)}")
