"""
places.py — Google Places API integration (stops) + Unsplash (destination photos).

Places API (New) is used for specific named stops (hotels, restaurants,
landmarks) where it returns accurate, place-specific photos.

Unsplash is used for the destination hero/gallery, because Places API
photos for a bare city/region query are often low-quality or blurry
(user-submitted snapshots), whereas Unsplash returns curated, high-res
travel photography — exactly what a hero banner needs.

Free tier (confirmed on Google's own pricing page, Oct 2026 — the old
$200/month credit this comment used to cite expired Feb 2025 and no
longer applies): Text Search Pro gets 5,000 free calls/month, Text Search
Enterprise only 1,000/month — see _FACT_FIELDS_MASK_SUFFIX_PRO/_ENTERPRISE
below for which call sites use which tier and why. Unsplash: 50 req/hour
on the free Demo tier — plenty for this use case.
"""

import os
import re
import time
import uuid
import requests

import database
from models import UnsplashAttribution
from ai_analyzer import _booking_affiliate_url, _expedia_affiliate_url, _hotel_checkin_checkout

PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "")

SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
UNSPLASH_SEARCH_URL = "https://api.unsplash.com/search/photos"

# Oct 2026 task 8 (revised after Gerry's cost review): the real-Google-facts
# fields that get appended to a call's own base field mask, split into two
# tiers because each one has a DIFFERENT free monthly quota, confirmed on
# Google's own pricing page (developers.google.com/maps/billing-and-pricing/
# pricing) — this was missed in the first pass, which lumped everything
# into one shared suffix and silently moved every video-route stop lookup
# onto the pricier tier:
#   Text Search Pro:        5,000 free calls/month, then $32/1000
#   Text Search Enterprise: 1,000 free calls/month, then $35/1000
#   (Enterprise+Atmosphere: 1,000 free/month, $40/1000 — still never used
#   here; neither suffix below requests editorialSummary)
# primaryTypeDisplayName/addressComponents/formattedAddress are Text Search
# PRO fields (confirmed against Google's Place Data Fields (New) docs,
# Text Search column) — adding them to a call that's already Pro-tier
# (photos/displayName/location/types) costs no SKU upgrade.
# rating/userRatingCount/priceLevel/regularOpeningHours are Text Search
# ENTERPRISE fields — requesting ANY one of them moves the WHOLE call to
# the Enterprise SKU (and its 5x-smaller free quota).
_FACT_FIELDS_MASK_SUFFIX_PRO = (
    ",places.primaryTypeDisplayName,places.addressComponents,places.formattedAddress"
)
# Adds the Enterprise-tier fields on top of the Pro ones above — Build Your
# Own Trip's attraction/hotel search (_ATTRACTION_SEARCH_FIELD_MASK,
# _HOTEL_SEARCH_FIELD_MASK below) already needed priceLevel/rating/
# userRatingCount for budget classification and curation BEFORE task 8
# existed, so those two calls were already on the Enterprise SKU — this
# suffix just keeps that unchanged while adding addressComponents/
# regularOpeningHours/formattedAddress to the same already-Enterprise call
# for free (no further SKU to move up to). Video-route stop lookups
# (_search_places, used by enrich_itinerary_with_photos) must NOT use this
# one — see _FACT_FIELDS_MASK_SUFFIX_PRO above for those.
_FACT_FIELDS_MASK_SUFFIX_ENTERPRISE = (
    _FACT_FIELDS_MASK_SUFFIX_PRO
    + ",places.rating,places.userRatingCount,places.priceLevel,places.regularOpeningHours"
)

# Same persistent Volume main.py's /admin/upload-image writes to (see
# IMAGES_DIR there) — reused here so a Places photo, once downloaded, is
# served by this backend forever instead of re-hitting Google. Computed
# from database.DATA_DIR rather than imported from main.py to avoid a
# circular import (main.py imports from this module).
IMAGES_DIR = database.DATA_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "https://getway-production.up.railway.app").rstrip("/")
_PLACE_PHOTO_CONTENT_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


def _build_photo_media_url(photo_name: str, max_width: int = 1600) -> str:
    return (
        f"https://places.googleapis.com/v1/{photo_name}/media"
        f"?maxWidthPx={max_width}&key={PLACES_API_KEY}&skipHttpRedirect=false"
    )


def _cache_photo_to_volume(media_url: str) -> str:
    """
    Downloads a Google Places photo's actual bytes right now and saves them
    onto our own persistent Volume (see IMAGES_DIR above) — the same one
    /admin/upload-image already writes to — turning Google's ephemeral,
    API-key-bearing media URL into a permanent one this backend serves
    itself.

    Real bug this fixes: _build_photo_url used to hand back that raw
    Google URL directly, and it got stored as-is in the database. Places
    photo `name` tokens are NOT permanent — confirmed live with curl that
    a stop photo fetched months ago now 400s with "The photo resource in
    the request is invalid. Please retrieve it from Places API
    endpoints." Every route already has stops whose photos have quietly
    died this way (frontend's imgFallback correctly swaps in the gradient
    placeholder, which is why this reads as "missing photo" rather than a
    hard error — see POST /admin/backfill-stop-photos in main.py for
    re-fixing those). This stops it from happening to any NEW route.

    Returns "" on any failure (network hiccup, unexpected content-type,
    disk write error) so the caller can fall back to the raw — still
    works right now, just not forever — Google URL instead of losing the
    photo outright.
    """
    try:
        resp = requests.get(media_url, timeout=10)
        if resp.status_code != 200 or not resp.content:
            return ""
        content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        extension = _PLACE_PHOTO_CONTENT_TYPES.get(content_type)
        if not extension:
            return ""
        filename = f"places_{uuid.uuid4().hex}{extension}"
        (IMAGES_DIR / filename).write_bytes(resp.content)
        return f"{BACKEND_BASE_URL}/uploads/{filename}"
    except (requests.RequestException, OSError) as e:
        print(f"[Places] Photo cache-to-volume failed (falling back to the raw Google URL): {e}")
        return ""


def _build_photo_url(photo_name: str, max_width: int = 1600) -> str:
    if not photo_name:
        return ""
    media_url = _build_photo_media_url(photo_name, max_width)
    return _cache_photo_to_volume(media_url) or media_url


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
                    # Oct 2026 task 8, revised after Gerry's cost review:
                    # expanded from photos/displayName/location only, so a
                    # confirmed VIDEO-ROUTE stop also carries real Google
                    # facts (type, neighbourhood, address) instead of
                    # nothing — see _FACT_FIELDS_MASK_SUFFIX_PRO above.
                    # Deliberately the PRO-only suffix, not the Enterprise
                    # one: this call already runs on EVERY stop of EVERY
                    # video route, and Enterprise's free quota (1,000/mo)
                    # is 5x smaller than Pro's (5,000/mo) — at current
                    # traffic that would start costing money after ~55 new
                    # routes/month, for rating/priceLevel/opening-hours data
                    # this function doesn't even return yet (deferred to a
                    # later task). Staying on Pro keeps video routes inside
                    # the free quota. No rating/userRatingCount/priceLevel/
                    # regularOpeningHours fields here — see _facts_from_place,
                    # which already defaults them to 0/""/[] when absent.
                    # places.types added Nov 2026 (task 14 route-quality
                    # pass): a Basic-tier field (free, no SKU change) —
                    # needed as the fallback source for place_type when
                    # Google's primaryTypeDisplayName comes back empty,
                    # which it does for a large share of real, confidently-
                    # matched places (villages/towns, streets, neighbourhoods,
                    # even some famous landmarks) — see
                    # _TYPE_DISPLAY_FALLBACK/_place_type_fallback below.
                    "X-Goog-FieldMask": "places.photos,places.displayName,places.location,places.types" + _FACT_FIELDS_MASK_SUFFIX_PRO,
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
# Oct 2026 task 8: + _FACT_FIELDS_MASK_SUFFIX_ENTERPRISE for type/address/
# area/hours (priceLevel/rating/userRatingCount were already being
# requested below, on their own, before task 8 existed — this call was
# already on the Enterprise SKU because of them, so adding
# addressComponents/regularOpeningHours/formattedAddress alongside costs no
# additional SKU tier; see _FACT_FIELDS_MASK_SUFFIX_ENTERPRISE's comment).
_ATTRACTION_SEARCH_FIELD_MASK = (
    "places.id,places.displayName,places.location,places.photos,"
    "places.types,places.businessStatus"
    + _FACT_FIELDS_MASK_SUFFIX_ENTERPRISE
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
# userRatingCount, businessStatus), now via the shared
# _FACT_FIELDS_MASK_SUFFIX_ENTERPRISE (task 8), which also brings
# addressComponents/primaryTypeDisplayName/regularOpeningHours a hotel
# recommendation can use the same as any stop. This call was already
# Enterprise-tier before task 8 (same reasoning as _ATTRACTION_SEARCH_
# FIELD_MASK above), so this adds no new SKU cost.
_HOTEL_SEARCH_FIELD_MASK = (
    "places.displayName,places.location,places.photos,places.businessStatus"
    + _FACT_FIELDS_MASK_SUFFIX_ENTERPRISE
)


def search_hotels_near(lat: float, lng: float, city: str, radius_meters: int = 4000,
                        max_results: int = 20, _retries: int = 2) -> list[dict]:
    """
    Geographically-anchored hotel search: a bare "hotels" Text Search
    biased toward (lat, lng) via Places API (New)'s locationBias, so
    results cluster around wherever the traveler's selected attractions
    actually are — not wherever Google's own city-center bias happens to
    land a plain "hotels in {city}" query. This is what makes the LOCKED
    LOGIC guarantee possible ("cheap" budget never means a worse location):
    trip_builder.pick_hotel() filters/ranks these results, but it can only
    pick from what's actually near the traveler's chosen stops.

    The query used to be f"hotels in {city}" — dropped the destination
    name in favor of a bare "hotels" so a REGIONAL destination (e.g.
    "Tuscany, Italy" for a multi-town trip) can't have its broad-region
    text relevance compete with locationBias instead of reinforcing it.
    Turned out NOT to be the actual fix for the far-hotel bug this was
    investigating (see trip_builder.pick_hotel's _HOTEL_PREFERRED_
    DISTANCE_KM comment for the real root cause and fix) — the raw
    results already included closer options, pick_hotel's ranking just
    wasn't weighing distance. Kept anyway since it's a real, low-risk
    improvement on its own reasoning, just not sufficient by itself.

    locationBias is a soft preference, not a hard filter — Google can still
    return something outside `radius_meters` if nothing better exists.
    Returns the raw places list with the field mask above; unclassified —
    see trip_builder.pick_hotel() for the 4.0+ rating floor and budget fit.
    """
    if not PLACES_API_KEY:
        return []
    query = "hotels"
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


def resolve_destination_place(name: str) -> dict | None:
    """
    v12 "Best this month" fix (Nov 2026): split out of what used to be
    destination_is_real_place's own inline call so the ONE Places lookup
    for a destination can now back TWO independent checks —
    destination_matches_place (unchanged logic, see below) AND
    destination_matches_region (new, see its own docstring) — without
    doubling the Places API call generate_month_calendar makes per
    candidate. Returns the raw place dict (photos/displayName/location/
    types/addressComponents/... per _search_places' field mask) for the
    best match, or None if nothing came back or the lookup itself failed.
    Callers that need both checks should call this ONCE, cache the
    result, and pass it to both — see generate_month_calendar's
    _resolve_place_cached.
    """
    name = (name or "").strip()
    if not name:
        return None
    try:
        results = _search_places(name, max_results=1)
    except Exception as e:
        print(f"[Places] resolve_destination_place('{name}') lookup failed: {type(e).__name__}: {e}")
        return None
    return results[0] if results else None


def destination_matches_place(name: str, place: dict | None) -> bool:
    """
    The name-overlap check that used to be destination_is_real_place's
    entire body (see that function below) — now takes an already-resolved
    `place` (from resolve_destination_place) instead of doing its own
    Places lookup, so a caller that also needs destination_matches_region
    for the same destination doesn't pay for two network calls.
    """
    if not place:
        return False
    display_name = (place.get("displayName") or {}).get("text", "").strip()
    if not display_name:
        return False
    return _names_plausibly_match(name, display_name)


def destination_is_real_place(name: str) -> bool:
    """
    v11 "Best this month" fix (Nov 2026): the AI month-calendar generator
    (ai_analyzer.generate_month_calendar) used to occasionally put an
    EVENT's own name in the destination field itself — e.g. "Thanksgiving,
    USA nationwide", "Dragon Boat Festival (China, East Asia)" — instead
    of a real, bookable place. Harmless-looking JSON, but broken on the
    homepage card (its photo lookup has nothing real to search for) and a
    factual-accuracy problem in its own right, found during the v10
    audit. Confirms `name` resolves to an actual Google Places result
    whose own name plausibly overlaps what was asked for — the same
    "is this really the place I asked for" check the stop-resolution
    pipeline already uses (_names_plausibly_match above). Cheap (1
    result, minimal field mask, via _search_places) and good enough: a
    genuine place reliably resolves to itself, while an event name either
    returns nothing from Places or returns something whose own name
    shares no real overlap with it.

    v12 (Nov 2026): now a thin wrapper around resolve_destination_place +
    destination_matches_place (split out so generate_month_calendar can
    share one Places lookup with the new destination_matches_region check
    below) — kept as its own function since nothing about calling this
    one function for "is this real" changed for any existing caller.
    """
    return destination_matches_place(name, resolve_destination_place(name))


# v12 code-level fix (Nov 2026) — found only by Gerry's manual read-through
# of a full v11 generation: a Europe-located, globally-famous event
# (Wimbledon/London, Edinburgh Festival Fringe, Notting Hill Carnival/
# London, Oktoberfest/Munich) got past every other v11 check — event_id/
# month/location were all genuinely valid per _VERIFIED_EVENTS, and the
# destination resolved fine in destination_is_real_place (London, Edinburgh
# and Munich are all real, independently-bookable places) — because
# nothing in the pipeline ever checked that the destination's own country
# actually falls inside the CONTINENT ai_analyzer.generate_month_calendar
# was generating for at the time (region is set per region-loop iteration,
# never derived from the destination itself — see that function's
# docstring). Per-region prompting alone doesn't reliably stop the model
# reaching for a universally well-known destination/event even when it's
# outside the region it was actually asked to populate.
#
# Deliberately conservative — see destination_matches_region's docstring
# for why an unrecognized or genuinely transcontinental country is let
# through rather than rejected: this is one more net for the specific
# region-bucket bug above, not a second "is this real" check, and the
# other v11 checks already carry the main real-vs-fake burden.
_COUNTRY_CODE_TO_REGION: dict[str, str] = {
    # Europe
    "GB": "Europe", "IE": "Europe", "FR": "Europe", "DE": "Europe", "ES": "Europe",
    "PT": "Europe", "IT": "Europe", "NL": "Europe", "BE": "Europe", "CH": "Europe",
    "AT": "Europe", "GR": "Europe", "PL": "Europe", "CZ": "Europe", "SK": "Europe",
    "HU": "Europe", "RO": "Europe", "BG": "Europe", "HR": "Europe", "SI": "Europe",
    "RS": "Europe", "DK": "Europe", "SE": "Europe", "NO": "Europe", "FI": "Europe",
    "IS": "Europe", "EE": "Europe", "LV": "Europe", "LT": "Europe", "UA": "Europe",
    "MT": "Europe", "CY": "Europe", "LU": "Europe", "MC": "Europe", "AD": "Europe",
    "SM": "Europe", "VA": "Europe", "LI": "Europe", "AL": "Europe", "MK": "Europe",
    "ME": "Europe", "BA": "Europe", "MD": "Europe",
    # Asia
    "JP": "Asia", "CN": "Asia", "KR": "Asia", "TH": "Asia", "VN": "Asia",
    "IN": "Asia", "ID": "Asia", "MY": "Asia", "SG": "Asia", "PH": "Asia",
    "KH": "Asia", "LA": "Asia", "MM": "Asia", "NP": "Asia", "BT": "Asia",
    "LK": "Asia", "BD": "Asia", "PK": "Asia", "MN": "Asia", "TW": "Asia",
    "HK": "Asia", "MO": "Asia", "IR": "Asia", "UZ": "Asia", "KZ": "Asia",
    "KG": "Asia", "TJ": "Asia", "TM": "Asia", "AE": "Asia", "SA": "Asia",
    "QA": "Asia", "IL": "Asia", "JO": "Asia", "MV": "Asia",
    # Africa
    "ZA": "Africa", "EG": "Africa", "MA": "Africa", "KE": "Africa", "TZ": "Africa",
    "RW": "Africa", "UG": "Africa", "ET": "Africa", "GH": "Africa", "NG": "Africa",
    "BJ": "Africa", "BW": "Africa", "NA": "Africa", "ZW": "Africa", "ZM": "Africa",
    "MG": "Africa", "TN": "Africa", "SN": "Africa", "CI": "Africa", "MU": "Africa",
    "SC": "Africa",
    # North America (incl. Central America/Caribbean, by common
    # travel-industry convention — matches how this app's own REGIONS
    # filter groups them, see index.html's REGIONS array)
    "US": "North America", "CA": "North America", "MX": "North America",
    "CR": "North America", "PA": "North America", "GT": "North America",
    "BZ": "North America", "HN": "North America", "NI": "North America",
    "SV": "North America", "CU": "North America", "JM": "North America",
    "DO": "North America", "BS": "North America", "BB": "North America",
    "TT": "North America", "PR": "North America", "AG": "North America",
    "GD": "North America", "LC": "North America", "VC": "North America",
    "KN": "North America", "DM": "North America",
    # South America
    "BR": "South America", "AR": "South America", "CL": "South America",
    "PE": "South America", "CO": "South America", "BO": "South America",
    "EC": "South America", "VE": "South America", "UY": "South America",
    "PY": "South America", "GY": "South America", "SR": "South America",
    # Oceania
    "AU": "Oceania", "NZ": "Oceania", "FJ": "Oceania", "PG": "Oceania",
    "VU": "Oceania", "WS": "Oceania", "TO": "Oceania", "FM": "Oceania",
    "PW": "Oceania", "KI": "Oceania", "SB": "Oceania", "NC": "Oceania",
    "PF": "Oceania", "GU": "Oceania",
}

# Countries genuinely split across two continents by common geographic
# convention (Europe/Asia), where picking a side would be arbitrary —
# never rejected on a region mismatch regardless of which region asked.
_AMBIGUOUS_TRANSCONTINENTAL = {"RU", "TR", "GE", "AZ", "AM"}

# The 6 region names this app's region filter actually uses (see
# index.html's REGIONS array / ai_analyzer._REGION_ORDER) — anything else
# passed as expected_region is treated as "can't check this", not a
# mismatch.
_KNOWN_REGIONS = {"Europe", "Asia", "Africa", "North America", "South America", "Oceania"}


def destination_matches_region(place: dict | None, expected_region: str) -> bool:
    """
    Returns False ONLY on a confirmed country/region mismatch (e.g. a
    London result when expected_region is "North America") — True for a
    genuine match, a missing/unresolved place, an unrecognized country
    code, or a transcontinental country in _AMBIGUOUS_TRANSCONTINENTAL.
    See the comment above _COUNTRY_CODE_TO_REGION for why this stays
    conservative rather than trying to be a second "is this real" check.

    Takes an already-resolved `place` dict (from resolve_destination_place)
    — no network call here, so it's free to call alongside
    destination_matches_place for the same cached lookup.
    """
    if not place or expected_region not in _KNOWN_REGIONS:
        return True
    country_code = ""
    for comp in place.get("addressComponents") or []:
        if "country" in (comp.get("types") or []):
            country_code = (comp.get("shortText") or "").strip().upper()
            break
    if not country_code or country_code in _AMBIGUOUS_TRANSCONTINENTAL:
        return True
    actual_region = _COUNTRY_CODE_TO_REGION.get(country_code)
    if not actual_region:
        return True
    return actual_region == expected_region


def destination_search_city(destination: str) -> str:
    """
    Best-effort city/region name to append to a stop's own name when
    searching Places or Unsplash (so a generic stop name like "Zócalo"
    doesn't get confused with a same-named place elsewhere), and the
    (city, stop name) key used for stop_cache lookups/writes — every call
    site that needs either MUST go through this one function, or cache
    reads and writes silently stop matching each other.

    For a normal single-destination route ("Vienna, Austria") this is
    just the part before the first comma/&/and — unchanged from the
    original logic this replaces (inlined identically at 7 call sites
    before Nov 2026's task 14 route-quality pass).

    For a multi-city or multi-leg route ("Mexico City to Cancún, Mexico",
    "Hawaii (Oahu & Maui), USA") that first-split heuristic used to grab a
    garbled fragment and glue it onto EVERY stop regardless of which leg
    it's actually on: "Mexico City to Cancún" appended to "Taxco" or
    "Zócalo" (nowhere near Mexico City OR Cancún specifically — confirmed
    live as the reason Zócalo's Places search came back with literally no
    match, despite being one of the most famous squares in the Americas);
    or even "Hawaii (Oahu" for "Hawaii (Oahu & Maui), USA" (the split
    regex matched the "&" inside the parens before it ever reached the
    comma). Detected via a standalone "to" or a "(" in the string; falls
    back to the country name after the LAST comma instead — less specific
    than a real city, but always geographically correct for every single
    stop on a multi-leg route, which beats a wrong or nonsensical one
    every time.
    """
    d = (destination or "").strip()
    is_multi_leg = bool(re.search(r"\bto\b", d, re.IGNORECASE)) or "(" in d
    if is_multi_leg and "," in d:
        return d.rsplit(",", 1)[-1].strip()
    return re.split(r"\s*(?:,|&|\band\b)\s*", d, maxsplit=1, flags=re.IGNORECASE)[0].strip()


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


def _empty_facts() -> dict:
    """A fresh all-empty facts dict — task 8's "Places has nothing to say
    about this place" answer. Always a new dict (never a shared/mutated
    default) since callers may fill it in piecemeal."""
    return {
        "place_type": "", "address": "", "area_label": "",
        "opening_hours_weekly": [], "rating": 0.0, "user_rating_count": 0,
        "price_level": "",
    }


def _place_type_display(place: dict) -> str:
    """Google's own human-readable category for a place, e.g. "Art museum"
    (Places' primaryTypeDisplayName) — never our own guess at a category."""
    return ((place.get("primaryTypeDisplayName") or {}).get("text") or "").strip()


# Nov 2026 (task 14 route-quality pass): primaryTypeDisplayName (above) is
# genuinely EMPTY for a large share of real, confidently-matched places —
# confirmed live across a 20-stop sample of "missing Google data" stops:
# 70% actually had a full formattedAddress (Places found them fine) but no
# display-name type at all — villages/towns (Qrendi, Frigiliana, Taxco),
# streets (Victoria Street, Champs-Élysées), neighbourhoods (Trastevere),
# and even a famous landmark (Casa Milà). Google's raw `types` array (a
# Basic-tier field — always returned, no extra SKU cost, just needed in the
# field mask — see _search_places) is far more reliable, just machine-coded
# rather than human-readable, so this maps the common ones a travel stop
# can plausibly be to a short display label. Checked in array order (a
# place's own types list leads with its most-specific type), first match
# wins. Deliberately NOT exhaustive — covers the categories that actually
# show up on a travel itinerary, with "Landmark" as the catch-all for the
# two broadest, near-universal types (tourist_attraction/point_of_interest)
# so a real, confirmed place essentially never has to come back with no
# type label at all.
_TYPE_DISPLAY_FALLBACK = {
    "museum": "Museum", "art_gallery": "Art gallery",
    "church": "Church", "hindu_temple": "Temple", "mosque": "Mosque",
    "synagogue": "Synagogue", "place_of_worship": "Place of worship",
    "park": "Park", "national_park": "National park",
    "natural_feature": "Natural landmark", "beach": "Beach",
    "zoo": "Zoo", "amusement_park": "Amusement park", "aquarium": "Aquarium",
    "shopping_mall": "Shopping", "market": "Market",
    "night_club": "Nightlife", "bar": "Bar", "cafe": "Café",
    "restaurant": "Restaurant", "bakery": "Bakery", "food": "Restaurant",
    "lodging": "Hotel", "campground": "Campground",
    "route": "Street", "neighborhood": "Neighbourhood",
    "sublocality": "Neighbourhood", "sublocality_level_1": "Neighbourhood",
    "locality": "Village/town", "administrative_area_level_3": "Village/town",
    "stadium": "Stadium", "plaza": "Square", "marina": "Marina",
    "spa": "Spa", "casino": "Casino",
    "tourist_attraction": "Landmark", "point_of_interest": "Landmark",
}


def _place_type_fallback(place: dict) -> str:
    """Fallback category label from Google's raw `types` array — see
    _TYPE_DISPLAY_FALLBACK above for why/when this is needed."""
    for t in place.get("types") or []:
        label = _TYPE_DISPLAY_FALLBACK.get(t)
        if label:
            return label
    return ""


def _area_label_from_components(place: dict) -> str:
    """
    Pulls a neighbourhood/area name out of Places' addressComponents —
    task 8: this used to be an AI guess (hotel stops only); now it's
    Google's own geocoded component, for every stop. Prefers a true
    "neighborhood" component; falls back to the broader "sublocality"
    level when a place has no finer neighbourhood data (both read as "the
    area" to a traveler, unlike "locality", which is the whole city).
    Returns "" when Places has neither.
    """
    components = place.get("addressComponents") or []
    for wanted in ("neighborhood", "sublocality", "sublocality_level_1"):
        for comp in components:
            if wanted in (comp.get("types") or []):
                return (comp.get("longText") or comp.get("shortText") or "").strip()
    return ""


def _opening_hours_weekly(place: dict) -> list[str]:
    """
    Google's own 7-line weekly schedule text (e.g. "Monday: 9:00 AM – 6:00
    PM"), verbatim — task 8. Deliberately NOT "open now" or "today's
    hours": those are only correct at the moment they're fetched and go
    stale the instant this gets cached/reused for a later visitor, so
    "open now"/"today" is computed by the frontend from this weekly
    schedule at render time instead, never stored.
    """
    hours = (place.get("regularOpeningHours") or {}).get("weekdayDescriptions") or []
    return [str(h).strip() for h in hours if str(h).strip()]


def _facts_from_place(place: dict) -> dict:
    """Bundles every task-8 Google-sourced fact for one matched place —
    see _empty_facts for the shape when nothing is known."""
    return {
        "place_type": _place_type_display(place) or _place_type_fallback(place),
        "address": (place.get("formattedAddress") or "").strip(),
        "area_label": _area_label_from_components(place),
        "opening_hours_weekly": _opening_hours_weekly(place),
        "rating": place.get("rating", 0) or 0.0,
        "user_rating_count": place.get("userRatingCount", 0) or 0,
        "price_level": place.get("priceLevel", "") or "",
    }


def _strip_parenthetical(name: str) -> str:
    """
    Strips a trailing "(...)" annotation from a place name, e.g.
    "Zócalo (Plaza de la Constitución)" -> "Zócalo". Used as a one-time
    retry in _get_place_photo_location_and_facts below.
    """
    return re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()


def _get_place_photo_location_and_facts(query: str, max_width: int = 800) -> tuple[str, tuple[float, float] | None, dict]:
    """
    Same name-matching logic as _get_place_photo_url, but also returns the
    matched place's (lat, lng) — used to populate Stop.lat/Stop.lng for the
    route Map view (index.html) — AND its task-8 facts dict (see
    _facts_from_place), all for free: everything rides along on the same
    Text Search response already fetched for the photo. Returns ("", None,
    _empty_facts()) when nothing confidently matches; returns ("", location,
    facts) when a confident name match has coordinates/facts but no usable
    photo — a stop can still get a map pin and real facts without a photo.

    Fallback (Nov 2026 — Zócalo, Mexico City): when the stop's name has a
    parenthetical annotation like "Zócalo (Plaza de la Constitución)" and
    the full form finds no confident match, retries ONCE with the
    annotation stripped. Google's Text Search can fail to resolve the
    combined form even though the bare name alone matches cleanly — this
    costs one extra Places call, only ever on an otherwise-failed lookup.
    """
    result = _get_place_photo_location_and_facts_core(query, max_width)
    _photo_url, location, facts = result
    if location is not None or any(facts.values()):
        return result  # already got a usable match — no need to retry

    query_core, _, rest = query.partition(",")
    if "(" not in query_core:
        return result  # nothing to strip, retrying would just repeat the same search

    stripped_core = _strip_parenthetical(query_core)
    if not stripped_core or stripped_core == query_core:
        return result

    retry_query = f"{stripped_core},{rest}" if rest else stripped_core
    print(f"[Places] No match for '{query}' — retrying without parenthetical as '{retry_query}'")
    return _get_place_photo_location_and_facts_core(retry_query, max_width)


def _get_place_photo_location_and_facts_core(query: str, max_width: int = 800) -> tuple[str, tuple[float, float] | None, dict]:
    """Original single-attempt body of _get_place_photo_location_and_facts — see that function's docstring."""
    places = _search_places(query, max_results=3)
    if not places:
        print(f"[Places] No places found for '{query}'")
        return "", None, _empty_facts()

    query_core = query.split(",")[0]  # drop the appended city for the name comparison
    for place in places:
        name = place.get("displayName", {}).get("text", "")
        if not _names_plausibly_match(query_core, name):
            continue
        location = _location_from_place(place)
        facts = _facts_from_place(place)
        photos = place.get("photos", [])
        if not photos:
            return "", location, facts
        candidates = photos[:5]
        landscape = [p for p in candidates if p.get("widthPx", 0) > p.get("heightPx", 0)]
        best = max(landscape or candidates, key=lambda p: p.get("widthPx", 0))
        return _build_photo_url(best.get("name", ""), max_width), location, facts

    print(f"[Places] No confident name match for '{query}' — skipping photo rather than risk a wrong one")
    return "", None, _empty_facts()


def _get_place_photo_and_location(query: str, max_width: int = 800) -> tuple[str, tuple[float, float] | None]:
    """
    Thin wrapper around _get_place_photo_location_and_facts for call sites
    that only need the photo+location (kept so nothing else has to
    change) — see that function for the real logic.
    """
    url, location, _facts = _get_place_photo_location_and_facts(query, max_width)
    return url, location


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


def best_fresh_unsplash_photo(query: str, used_ids: set | None = None) -> tuple[str, UnsplashAttribution | None]:
    """
    Returns (url, attribution) for the best-liked Unsplash result for
    `query` that isn't already in `used_ids`, or ("", None) if nothing
    came back. Shared by enrich_itinerary_with_photos (which passes a set
    it tracks across every stop in one itinerary, so the same photo never
    shows twice on one route page) and trip_builder.hotel_to_recommendation_dict
    (a one-off lookup — pass None, a fresh set is used).
    """
    if used_ids is None:
        used_ids = set()
    for c in sorted(_unsplash_candidates(query, per_page=10), key=lambda r: r.get("likes", 0), reverse=True):
        cid, url = c.get("id"), c.get("urls", {}).get("regular")
        if url and cid not in used_ids:
            used_ids.add(cid)
            _trigger_unsplash_download(c)
            return url, _attribution_from_candidate(c)
    return "", None


def _get_destination_gallery_unsplash(destination: str, count: int = 5) -> list[dict]:
    """
    Returns up to `count` curated, high-resolution travel photo entries for
    a destination from Unsplash, each as {"url": ..., "likes": ...}. Uses
    several distinct queries (rather than one broad query) so the gallery
    shows varied shots instead of several near-duplicate frames from the
    same photo session. Falls back to an empty list if no key is
    configured or all requests fail.

    Cached by (destination, count) — see database.get_destination_gallery_cache.
    Real bug this fixes: called fresh on EVERY /trip/build with no caching
    at all, burning 2-5+ Unsplash queries each time against a free-tier
    50-requests/hour cap shared with every other Unsplash call in the app —
    several Build Your Own Trip previews for popular destinations in the
    same hour could exhaust it, silently starving whichever photo fetch
    ran later in that same request (see get_car_rental_photo_unsplash).
    A destination's gallery doesn't change traveler-to-traveler, so this is
    safe to share across everyone building a trip to the same place.
    """
    cached = database.get_destination_gallery_cache(destination, count)
    if cached is not None:
        for p in cached:
            p["attribution"] = UnsplashAttribution(**p["attribution"]) if p.get("attribution") else None
        return cached

    if not UNSPLASH_ACCESS_KEY:
        print("[Unsplash] No API key — skipping destination gallery")
        return []

    # Multi-city destinations (e.g. "Cairo & Luxor", "Rome and Florence")
    # need to be split down to a single clean city name — querying Unsplash
    # with the full compound string returns few or no results, which starved
    # the gallery down to 0-1 photos and hid the thumbnail strip entirely.
    city = destination_search_city(destination)

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

    if not photos:
        # Last resort: Unsplash returned nothing at all — either a
        # transient rate-limit/network failure, or a genuinely thin
        # Unsplash library for this destination. Confirmed live: a fresh
        # New York build came back with a fully empty gallery this way,
        # leaving the route with no hero photo and the homepage's
        # "Hotels in X" banner with no photo either (it reuses this same
        # gallery). A destination should never end up completely
        # photo-less, so fall back to a Google Places Text Search for the
        # destination itself — the same photo source already proven
        # reliable for hotels/stops elsewhere in this file. Tried
        # Wikipedia's and Wikivoyage's REST summary APIs here first
        # (free, no key) but both frequently returned a flag, a map, or
        # an unrelated building instead of a real destination photo for
        # exactly the big, well-known places most likely to be searched
        # (confirmed live: "Norway" → the Norwegian flag icon, "New York
        # City" → a district map graphic) — worse than showing nothing.
        if PLACES_API_KEY:
            fallback_candidates = _search_places(destination, max_results=1)
            fallback_url = photo_url_from_places_photos(fallback_candidates[0].get("photos", [])) if fallback_candidates else ""
            if fallback_url:
                photos = [{"url": fallback_url, "likes": 0, "attribution": None}]
                print(f"[Unsplash] Gallery for '{destination}': 0 Unsplash photos — used a Google Places fallback instead")

    print(f"[Unsplash] Gallery for '{destination}': {len(photos)} photos (sampled {len(queries_to_try)} queries, {len(candidate_pool)} candidates)")
    serializable = [
        {"url": p["url"], "likes": p["likes"], "attribution": p["attribution"].model_dump() if p["attribution"] else None}
        for p in photos
    ]
    # Short TTL when nothing came back — same reasoning as
    # get_car_rental_photo_unsplash's cache: an empty result here could be
    # a genuine "no good photos" OR a transient rate-limit/network failure,
    # and a normal 30-day TTL would lock in the latter as if it were the
    # former.
    database.save_destination_gallery_cache(destination, count, serializable, ttl_days=30 if photos else 0.25)
    return photos


def get_car_rental_photo_unsplash(destination: str) -> tuple[str, UnsplashAttribution | None]:
    """
    One themed photo (a car/road scene actually IN the destination, not a
    generic stock car) for the "Rent a car in {city}" banner — same
    Unsplash-search-and-attribute pattern as _get_destination_gallery_unsplash,
    just with driving-specific query terms instead of generic travel ones.
    Real bug this fixes: models.Itinerary never had a car_rental_photo_url
    field at all, so that banner's <img> always had an empty src and fell
    straight to the plain icon+gradient placeholder — every single time,
    for every trip, regardless of destination. Returns ("", None) if
    nothing usable is found (no key configured, or all queries came up
    empty) — the frontend's existing icon+gradient fallback still applies.

    Cached by destination — see database.get/save_car_rental_photo_cache.
    This is the LAST Unsplash fetch in /trip/build's sequence (after the
    hero/gallery fetch above, which alone can burn 2-5+ queries), so
    without caching it's also the first one starved whenever a burst of
    Build Your Own Trip previews pushes the free-tier 50-requests/hour cap
    — confirmed live: a Norway build got a real hero/gallery photo but an
    empty car_rental_photo_url.
    """
    cached = database.get_car_rental_photo_cache(destination)
    if cached is not None:
        attribution = UnsplashAttribution(**cached["attribution"]) if cached.get("attribution") else None
        return cached["url"], attribution

    city = destination_search_city(destination)
    queries = [f"{city} scenic road drive", f"{city} road trip", f"{city} countryside road"]
    for query in queries:
        candidates = sorted(_unsplash_candidates(query, per_page=6), key=lambda r: r.get("likes", 0), reverse=True)
        if candidates:
            best = candidates[0]
            url = best.get("urls", {}).get("regular", "")
            if url:
                _trigger_unsplash_download(best)
                attribution = _attribution_from_candidate(best)
                database.save_car_rental_photo_cache(destination, url, attribution.model_dump())
                return url, attribution
    print(f"[Unsplash] No car rental photo found for '{destination}'")
    # Short TTL for a "nothing found" result — _unsplash_candidates returns
    # [] both for a genuine no-match AND for a transient failure (rate
    # limit, network error, timeout), so a normal 30-day TTL risks locking
    # in a rate-limit blip as if it were a permanent "no good photo exists"
    # — the exact bug this caching was added to fix. A few hours is enough
    # to stop a retry storm without outliving the rate-limit window.
    database.save_car_rental_photo_cache(destination, "", None, ttl_days=0.25)
    return "", None


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
    # count=5, not 1: the route page's "getting to know [destination]"
    # intro section shows a swipeable photo gallery. Safe to raise now
    # (this used to be kept at 1 to conserve Unsplash's free-tier quota)
    # because this function is cached by (destination, count) — see
    # database.get/save_destination_gallery_cache.
    gallery = _get_destination_gallery_unsplash(itinerary.destination, count=5)
    itinerary.gallery_photo_urls = [p["url"] for p in gallery]
    itinerary.gallery_attributions = [p["attribution"] for p in gallery]
    hero = max(gallery, key=lambda p: p["likes"]) if gallery else None
    itinerary.hero_photo_url = hero["url"] if hero else ""
    itinerary.hero_attribution = hero["attribution"] if hero else None
    print(f"[Photos] Hero: {itinerary.destination} → {bool(itinerary.hero_photo_url)}")

    # Single primary city (e.g. "Cairo" from "Cairo & Luxor") — same
    # derivation used for the gallery above (destination_search_city).
    # Deliberately not the full multi-city string: appending "Cairo &
    # Luxor" to every stop's search query would bias a Luxor stop's
    # photo/Maps search toward Cairo just because it shares the trip. For
    # a multi-leg route (" to " or "(" in the destination), falls back to
    # the country name instead of a garbled city fragment — see that
    # function's docstring. The AI usually already writes the correct
    # city into stop.name itself (e.g. "Egyptian Museum, Cairo") — the
    # fallback below only kicks in when it didn't.
    city = destination_search_city(itinerary.destination)
    used_ids: set[str] = set()  # avoid repeating a photo across stop cards

    def _best_fresh_unsplash(query: str) -> tuple[str, dict | None]:
        return best_fresh_unsplash_photo(query, used_ids)

    for day in itinerary.days:
        for stop in day.stops:
            is_specific = getattr(stop, "is_specific_name", True)
            name_query = stop.name if city.lower() in stop.name.lower() else f"{stop.name}, {city}"

            # Nov 2026 (task 14): a generic activity description ("Snorkelling
            # with sea turtles, Oahu", "ATV Ride Experience, Ubud") was never
            # going to match a Places business listing — is_specific already
            # being False means the block below correctly skips the Places
            # call, but place_type stayed blank forever since nothing else
            # set it. "Activity" here is a plain hardcoded label, not a
            # Places fact — the small stop-facts line has something to show
            # instead of nothing, same label CAT_CONFIG already uses for the
            # category badge above it. The GetYourGuide booking button
            # (frontend, category === 'activity') doesn't depend on this.
            if stop.category == "activity" and not is_specific:
                stop.place_type = "Activity"

            # Reuse a photo/address/facts already saved from a PREVIOUS
            # route for this exact city+stop name (e.g. the Colosseum
            # showing up again in a second Rome route) — skips the
            # Places/Unsplash lookup entirely, which both saves the API
            # call and means Gerry's earlier manual fix for this stop
            # carries over automatically instead of needing to be redone
            # every time.
            cached = database.get_cached_stop(city, stop.name)
            if cached:
                stop.photo_url = cached["photo_url"]
                if cached.get("maps_url_override"):
                    stop.maps_url_override = cached["maps_url_override"]
                if cached.get("lat") is not None and cached.get("lng") is not None:
                    stop.lat = cached["lat"]
                    stop.lng = cached["lng"]
                # Task 8 facts (place_type/address/area_label/opening hours/
                # rating/price_level) — empty strings/lists/zeros on rows
                # saved before this existed, which is fine: the frontend
                # already treats an empty fact as "don't show this field".
                stop.place_type = cached.get("place_type", "") or ""
                stop.address = cached.get("address", "") or ""
                stop.area_label = cached.get("area_label", "") or ""
                stop.opening_hours_weekly = cached.get("opening_hours_weekly") or []
                stop.rating = cached.get("rating", 0.0) or 0.0
                stop.user_rating_count = cached.get("user_rating_count", 0) or 0
                stop.price_level = cached.get("price_level", "") or ""
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
                        stop.photo_url, location, facts = _get_place_photo_location_and_facts(f"{real_name}, {city}")
                        if location:
                            stop.lat, stop.lng = location
                        stop.place_type = facts["place_type"]
                        stop.address = facts["address"]
                        stop.area_label = facts["area_label"]
                        stop.opening_hours_weekly = facts["opening_hours_weekly"]
                        stop.rating = facts["rating"]
                        stop.user_rating_count = facts["user_rating_count"]
                        stop.price_level = facts["price_level"]
                        booking_query = f"{real_name}, {city}"
                        # Task 7: same dated-search rule as every other
                        # Booking/Expedia link in the app — see
                        # ai_analyzer._hotel_checkin_checkout.
                        checkin, checkout = _hotel_checkin_checkout(getattr(itinerary, "duration", ""))
                        stop.booking_url = _booking_affiliate_url(booking_query, checkin, checkout)
                        stop.expedia_url = _expedia_affiliate_url(booking_query, checkin, checkout)

            if is_specific and PLACES_API_KEY:
                # Don't double up the city if the AI already wrote it into
                # the name (e.g. "Nile Corniche, Cairo") — searching
                # "Nile Corniche, Cairo, Cairo & Luxor" is redundant and
                # doesn't help Places match the right place.
                stop.photo_url, location, facts = _get_place_photo_location_and_facts(name_query)  # Places photo — no Unsplash attribution needed
                if location:
                    stop.lat, stop.lng = location
                # Task 8: facts (place_type/address/area_label/opening
                # hours/rating/price_level) come ONLY from Places, never
                # from the AI — see _facts_from_place. Left at their
                # Stop-default empty values when Places had no confident
                # match (facts is _empty_facts() in that case).
                stop.place_type = facts["place_type"]
                stop.address = facts["address"]
                stop.area_label = facts["area_label"]
                stop.opening_hours_weekly = facts["opening_hours_weekly"]
                stop.rating = facts["rating"]
                stop.user_rating_count = facts["user_rating_count"]
                stop.price_level = facts["price_level"]
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


def backfill_stop_facts(itinerary) -> int:
    """
    Task 8 backfill (Nov 2026): force-refreshes every stop's Places facts
    (place_type/address/area_label/opening_hours_weekly/rating/price_level)
    on a route that was generated BEFORE task 8 existed.

    Deliberately does NOT go through enrich_itinerary_with_photos' normal
    database.get_cached_stop() short-circuit. That cache is keyed by
    (city, stop name) and almost every stop on an old route already has a
    row there from its ORIGINAL generation — but that row was written
    before facts existed, so its place_type/address/etc. are all empty.
    Trusting the cache here would hit that empty row on every single stop
    and silently leave the backfill a no-op. So this always does a fresh
    Places Text Search per stop (same Pro-tier mask as _search_places —
    see _FACT_FIELDS_MASK_SUFFIX_PRO) and then writes the result back into
    the cache (upsert_stop_cache) so every OTHER route sharing this exact
    stop (e.g. the Colosseum across several Rome routes) picks up the same
    facts for free the next time IT loads, without its own Places call.

    Touches facts, and now ALWAYS lat/lng too when a fresh confident match
    has coordinates — Nov 2026 fix: this used to only fill lat/lng when
    the stop didn't already have ANY (`stop.lat is None or stop.lng is
    None`), which quietly protected a stop's OLD, possibly-wrong
    coordinates from ever being corrected by a later refresh — confirmed
    live as the cause of a real bug: "Cafe Bali" (a Bali route) had a
    correct, freshly-refreshed Bali address but kept stale coordinates
    pointing to Indiana, USA, 15,998 km away, because they were already
    non-null from an earlier bad match. refresh_places=true (the only
    caller of this function) means "trust this fresh Places data" — there
    is no separate manual lat/lng override field (unlike
    maps_url_override, which this still never touches) for this to
    clobber. Never touches photo_url either, so a photo Gerry already
    manually fixed for this stop is left alone.

    Activity-category stops with no confirmed specific name (e.g.
    "Snorkelling with sea turtles, Oahu", "ATV Ride Experience, Ubud") are
    skipped entirely — Places has nothing to match a generic activity
    description against, so every call for one of these was a guaranteed
    paid miss. place_type is set to the plain "Activity" label instead
    (matches enrich_itinerary_with_photos' new-route behavior — see
    there), and the frontend already has a working GetYourGuide button
    for this category regardless of place_type.

    Returns how many stops got at least one new fact (Places had no
    confident match for the rest, which is fine — they just keep their
    empty defaults, same as any other stop Places can't resolve).
    """
    filled = 0
    city = destination_search_city(itinerary.destination)
    for day in itinerary.days:
        for stop in day.stops:
            if stop.category == "activity" and not getattr(stop, "is_specific_name", True):
                stop.place_type = stop.place_type or "Activity"
                continue
            name_query = stop.name if city.lower() in stop.name.lower() else f"{stop.name}, {city}"
            photo_url, location, facts = _get_place_photo_location_and_facts(name_query)
            if not any(facts.values()):
                print(f"[Backfill] No Places match for '{stop.name}' in {city}")
                continue
            stop.place_type = facts["place_type"]
            stop.address = facts["address"]
            stop.area_label = facts["area_label"]
            stop.opening_hours_weekly = facts["opening_hours_weekly"]
            stop.rating = facts["rating"]
            stop.user_rating_count = facts["user_rating_count"]
            stop.price_level = facts["price_level"]
            if location:
                stop.lat, stop.lng = location
            database.upsert_stop_cache(
                city, stop.name, stop.photo_url or photo_url,
                maps_url_override=stop.maps_url_override or "",
                lat=stop.lat, lng=stop.lng, facts=facts,
            )
            filled += 1
    return filled


def apply_cached_stop_facts(itinerary) -> int:
    """
    Nov 2026: reads each stop's Places facts back out of stop_cache instead
    of calling the Places API — for when backfill_stop_facts() already ran
    on this route (so stop_cache is already fresh, since that function
    always writes its results there even on a dry run) and only the AI
    description needs regenerating. Costs nothing: no network call, just a
    local DB read per stop. Returns NO row (or a row with no photo_url —
    see database.get_cached_stop) for a stop stop_cache never backfilled,
    which just leaves that stop's facts as they already were — same
    behavior as backfill_stop_facts() skipping a stop Places had no match
    for, so callers can treat the two functions interchangeably.

    Does NOT touch lat/lng (those only matter for the map/distance view,
    already handled by backfill_stop_facts the first time) and does not
    write anything back — purely a read.
    """
    filled = 0
    city = destination_search_city(itinerary.destination)
    for day in itinerary.days:
        for stop in day.stops:
            cached = database.get_cached_stop(city, stop.name)
            if not cached or not any([cached["place_type"], cached["address"], cached["area_label"]]):
                continue
            stop.place_type = cached["place_type"]
            stop.address = cached["address"]
            stop.area_label = cached["area_label"]
            stop.opening_hours_weekly = cached["opening_hours_weekly"]
            stop.rating = cached["rating"]
            stop.user_rating_count = cached["user_rating_count"]
            stop.price_level = cached["price_level"]
            filled += 1
    return filled
