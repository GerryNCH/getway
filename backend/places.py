"""
places.py — Google Places API integration.
Fetches one representative photo URL per stop name.
Uses Places API (New) Text Search + photo endpoint.
Free tier: $200/month credit → ~5000 photo lookups free.
"""

import os
import requests

PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")

SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
PHOTO_URL  = "https://places.googleapis.com/v1/{name}/media"


def _get_place_photo_url(query: str, max_width: int = 800) -> str:
    """
    Given a search query (e.g. "Cafe 67 Rome"), returns a Google Places
    photo URL or empty string if nothing found / API key missing.
    """
    if not PLACES_API_KEY:
        return ""

    try:
        # Step 1: Text search to get place + first photo reference
        resp = requests.post(
            SEARCH_URL,
            json={"textQuery": query, "maxResultCount": 1},
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": PLACES_API_KEY,
                "X-Goog-FieldMask": "places.photos",
            },
            timeout=5,
        )
        data = resp.json()
        places = data.get("places", [])
        if not places:
            return ""

        photos = places[0].get("photos", [])
        if not photos:
            return ""

        photo_name = photos[0].get("name", "")
        if not photo_name:
            return ""

        # Step 2: Build the photo media URL
        photo_url = (
            f"https://places.googleapis.com/v1/{photo_name}/media"
            f"?maxWidthPx={max_width}&key={PLACES_API_KEY}&skipHttpRedirect=false"
        )
        return photo_url

    except Exception as e:
        print(f"[Places] Error fetching photo for '{query}': {e}")
        return ""


def enrich_itinerary_with_photos(itinerary) -> None:
    """
    Mutates the itinerary in-place: adds photo_url to every stop
    and sets hero_photo_url from the destination name.
    """
    if not PLACES_API_KEY:
        print("[Places] No API key — skipping photo enrichment")
        return

    # Hero photo for the destination
    itinerary.hero_photo_url = _get_place_photo_url(itinerary.destination)
    print(f"[Places] Hero: {itinerary.destination} → {bool(itinerary.hero_photo_url)}")

    # Photo for each stop
    for day in itinerary.days:
        for stop in day.stops:
            stop.photo_url = _get_place_photo_url(stop.name)
            print(f"[Places] Stop: {stop.name} → {bool(stop.photo_url)}")
