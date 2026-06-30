"""
places.py — Google Places API integration.
Fetches photo URLs per stop name, and a gallery of photos for the destination hero.
Uses Places API (New) Text Search + photo endpoint.
Free tier: $200/month credit → ~5000 photo lookups free.
"""

import os
import requests

PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")

SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"


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


def _get_destination_gallery(destination: str, count: int = 5) -> list[str]:
    """
    Builds a gallery of `count` high-quality photo URLs for a destination.
    Strategy: query several landmark/skyline angles and pool results, so the
    first photo (used as hero) is a strong, recognisable shot rather than
    whatever a single generic city-name search happens to return first.
    """
    photo_urls: list[str] = []

    queries = [
        f"top tourist attractions in {destination}",
        f"{destination} iconic skyline",
        f"{destination} historic landmark",
        destination,
    ]

    for query in queries:
        if len(photo_urls) >= count:
            break
        places = _search_places(query, max_results=count)
        for place in places:
            if len(photo_urls) >= count:
                break
            photos = place.get("photos", [])
            if photos:
                url = _build_photo_url(photos[0].get("name", ""))
                if url not in photo_urls:
                    photo_urls.append(url)

    print(f"[Places] Gallery for '{destination}': {len(photo_urls)} photos")
    return photo_urls


def enrich_itinerary_with_photos(itinerary) -> None:
    """
    Mutates the itinerary in-place:
      - adds photo_url to every stop
      - sets hero_photo_url (best single shot) and gallery_photo_urls (4-5 shots)
    """
    if not PLACES_API_KEY:
        print("[Places] No API key — skipping photo enrichment")
        return

    # Destination gallery — first photo doubles as the hero (best quality first)
    gallery = _get_destination_gallery(itinerary.destination, count=5)
    itinerary.gallery_photo_urls = gallery
    itinerary.hero_photo_url = gallery[0] if gallery else ""
    print(f"[Places] Hero: {itinerary.destination} → {bool(itinerary.hero_photo_url)}")

    # Photo for each stop
    for day in itinerary.days:
        for stop in day.stops:
            stop.photo_url = _get_place_photo_url(stop.name)
            print(f"[Places] Stop: {stop.name} → {bool(stop.photo_url)}")
