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
        if not photos:
            continue
        # Google returns photos in whatever order it ranks them internally
        # — often a random guest close-up (a boat passing in the distance,
        # a hallway) rather than a representative exterior/room shot.
        # Preferring a landscape-oriented, higher-resolution photo among
        # the first several is a free, cheap signal that tends to favor an
        # actual establishing shot over a narrow detail crop.
        candidates = photos[:5]
        landscape = [p for p in candidates if p.get("widthPx", 0) > p.get("heightPx", 0)]
        best = max(landscape or candidates, key=lambda p: p.get("widthPx", 0))
        return _build_photo_url(best.get("name", ""), max_width)

    print(f"[Places] No confident name match for '{query}' — skipping photo rather than risk a wrong one")
    return ""


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
                        stop.photo_url = _get_place_photo_url(f"{real_name}, {city}")
                        booking_query = f"{real_name}, {city}"
                        stop.booking_url = _booking_affiliate_url(booking_query)
                        stop.expedia_url = _expedia_affiliate_url(booking_query)

            if is_specific and PLACES_API_KEY:
                # Don't double up the city if the AI already wrote it into
                # the name (e.g. "Nile Corniche, Cairo") — searching
                # "Nile Corniche, Cairo, Cairo & Luxor" is redundant and
                # doesn't help Places match the right place.
                stop.photo_url = _get_place_photo_url(name_query)  # Places photo — no Unsplash attribution needed
                print(f"[Places] Stop: {name_query} → {bool(stop.photo_url)}")

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
