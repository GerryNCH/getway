"""
ai_analyzer.py — Multimodal Claude analysis.

Sends extracted video frames to Claude Sonnet with a carefully crafted prompt
that handles the "unnamed restaurant" problem:
  - Reads on-screen text (menus, signs, subtitles)
  - Recognises landmarks from visual appearance
  - Identifies logos and branded interiors
  - Infers location from context clues (beach type, architecture, language)
"""

import base64
import json
import re
import time
import urllib.parse
from datetime import date, timedelta

import anthropic
from models import Itinerary

_client = anthropic.Anthropic()

# Claude Sonnet 4.6 standard rate, $ per million tokens (verified July 2026 —
# update these two numbers if Anthropic changes pricing or analyse_frames'
# model string below is changed to a different model).
_SONNET_INPUT_PER_MTOK = 3.00
_SONNET_OUTPUT_PER_MTOK = 15.00

# Claude Haiku 4.5 standard rate, $ per million tokens — same figures used
# in quality_check.py and troll_filter.py. Used here by
# generate_grounded_descriptions() (task 8's fact-grounded stop
# descriptions) and generate_travel_tips()/generate_trip_summary(), never
# by the main Sonnet pass.
_HAIKU_INPUT_PER_MTOK = 1.00
_HAIKU_OUTPUT_PER_MTOK = 5.00

# ── Affiliate link config (Phase 2) ──────────────────────────────────────────
# CJ Affiliate account — Gerry's Publisher ID + Booking.com's Advertiser/Link
# ID. Booking.com North America approval covers the deep-link format below.
_CJ_PUBLISHER_ID = "101819605"
_BOOKING_LINK_ID = "17293132"
_CJ_BASE_URL = f"https://www.anrdoezrs.net/click-{_CJ_PUBLISHER_ID}-{_BOOKING_LINK_ID}"


# Task 7: Booking/Expedia links must carry real dates + occupancy, not
# just the destination — confirmed live (2026-10-29, via the browser
# tool) against both booking.com/searchresults.html (checkin/checkout=
# YYYY-MM-DD, group_adults, no_rooms, group_children) and
# expedia.com/Hotel-Search (startDate/endDate=YYYY-MM-DD, adults, rooms) —
# both resolved to real, correctly-dated/occupied search results, not a
# guess. Gerry approved "30 days from now + the route's own duration" for
# the date range, and 2 adults/1 room as the fixed occupancy (no per-
# traveler people count exists in this flow to use instead).
_HOTEL_SEARCH_LEAD_DAYS = 30


def _hotel_checkin_checkout_for_nights(nights: int) -> tuple[str, str]:
    """
    Returns (checkin, checkout) as YYYY-MM-DD strings: checkin is
    _HOTEL_SEARCH_LEAD_DAYS from today, checkout is that plus `nights`
    (at least 1). Use this directly when the caller already has a real
    integer day count (e.g. main.py's /trip/build, TripBuildRequest.days)
    rather than going through _hotel_checkin_checkout's string parsing.
    """
    nights = max(1, int(nights))
    checkin = date.today() + timedelta(days=_HOTEL_SEARCH_LEAD_DAYS)
    checkout = checkin + timedelta(days=nights)
    return checkin.isoformat(), checkout.isoformat()


def _hotel_checkin_checkout(duration: str) -> tuple[str, str]:
    """
    Same as _hotel_checkin_checkout_for_nights, but `nights` is parsed
    from a duration STRING (e.g. Itinerary.duration, "5 days" — the
    leading integer) instead of a known integer — for callers that only
    have that string (video-extracted routes; places.py's hotel-upgrade
    branch). Defaults to 3 nights if it can't be parsed, same fallback
    depth as other best-effort parses in this file.
    """
    match = re.search(r"\d+", duration or "")
    nights = int(match.group()) if match else 3
    return _hotel_checkin_checkout_for_nights(nights)


def _booking_affiliate_url(search_query: str, checkin: str = "", checkout: str = "") -> str:
    """
    Booking.com search-results URL, wrapped in the CJ affiliate tracking
    link so hotel clicks are attributed to Gerry's account.
    NOTE: the real search path is /searchresults.html, not /search.html.

    checkin/checkout (YYYY-MM-DD, both or neither — see
    _hotel_checkin_checkout): when given, also fixes occupancy at 2
    adults / 1 room / 0 children, confirmed live to produce a real dated
    search rather than Booking's default "tonight" search.
    """
    params = {"ss": search_query}
    if checkin and checkout:
        params.update({
            "checkin": checkin,
            "checkout": checkout,
            "group_adults": "2",
            "no_rooms": "1",
            "group_children": "0",
        })
    target = f"https://www.booking.com/searchresults.html?{urllib.parse.urlencode(params)}"
    return f"{_CJ_BASE_URL}?url={urllib.parse.quote(target, safe='')}"


# Expedia Group Travel Creator Program — approved July 2026. camref/
# creativeref/adref are the fixed identifiers tied to Gerry's account and
# this generated link (confirmed via their Link Builder tool, not guessed);
# only `landingPage` changes per hotel/destination search.
_EXPEDIA_CAMREF = "1110lK3nQ"
_EXPEDIA_CREATIVEREF = "1100l68075"
_EXPEDIA_ADREF = "PZTaSwOiKr"


def _expedia_affiliate_url(search_query: str, checkin: str = "", checkout: str = "") -> str:
    """
    Expedia Hotel-Search URL, wrapped in Gerry's Expedia Group affiliate
    link (via their Travel Creator Program) so hotel clicks are attributed
    to his account.

    checkin/checkout (YYYY-MM-DD, both or neither — see
    _hotel_checkin_checkout): when given, also fixes occupancy at 2
    adults / 1 room, confirmed live to produce a real dated search.
    """
    target_params = {"destination": search_query}
    if checkin and checkout:
        target_params.update({"startDate": checkin, "endDate": checkout, "adults": "2", "rooms": "1"})
    target = f"https://www.expedia.com/Hotel-Search?{urllib.parse.urlencode(target_params)}"
    params = {
        "siteid": "1",
        "landingPage": target,
        "camref": _EXPEDIA_CAMREF,
        "creativeref": _EXPEDIA_CREATIVEREF,
        "adref": _EXPEDIA_ADREF,
    }
    return f"https://expedia.com/affiliate?{urllib.parse.urlencode(params)}"


def _google_maps_search_url(search_query: str) -> str:
    """
    Temporary restaurant link until the TheFork affiliate application is
    approved — a plain Google Maps search, not affiliate-tracked.
    """
    return f"https://www.google.com/maps/search/{urllib.parse.quote_plus(search_query)}"


# GetYourGuide affiliate program — approved July 2026. partner_id and the
# utm_medium value are confirmed from a real link generated via Partner
# Dashboard → "Create a link" (not guessed). That tool only generates
# links to a specific activity page (e.g. .../madeira-skywalk-...-t225105/),
# and we can't know a specific activity's exact page slug/ID for an
# arbitrary stop — so this uses GetYourGuide's standard search page (/s/)
# instead, with the same confirmed affiliate params appended. Affiliate
# tracking params work the same way on any getyourguide.com page (they set
# a referral cookie, not a page-specific token), so this should track
# correctly, but it's worth spot-checking once real traffic comes through.
_GYG_PARTNER_ID = "UC97KXN"


def _getyourguide_affiliate_url(search_query: str) -> str:
    """GetYourGuide search-results URL with Gerry's affiliate params attached."""
    return (
        f"https://www.getyourguide.com/s/?q={urllib.parse.quote_plus(search_query)}"
        f"&partner_id={_GYG_PARTNER_ID}&utm_medium=online_publisher"
    )


def _attach_booking_urls(data: dict) -> dict:
    """
    Fills in `booking_url` (Booking.com, GetYourGuide, or Maps) and
    `expedia_url` (hotels only) for every hotel/food/activity stop, in
    place, before the JSON is turned into an Itinerary:
      - hotel    → Booking.com search wrapped in the CJ affiliate link,
        PLUS an Expedia affiliate search as a second option. Uses the
        exact stop name when the AI confirmed it (`is_specific_name`),
        otherwise falls back to just the destination city — same rule the
        frontend already applies for its own "unconfirmed hotel" fallback.
      - food     → Google Maps search (name + city) as a stand-in until
        TheFork affiliate is live.
      - activity/sight → GetYourGuide search wrapped in Gerry's affiliate
        params — sights (museums, landmarks) often need advance tickets
        too, not just tour-style activities.
    Any other category is left untouched.
    """
    destination = data.get("destination", "")
    # Nov 2026 (task 14): a plain first-comma split breaks the same way
    # places.destination_search_city's docstring describes for a
    # multi-leg/multi-island destination ("Mexico City to Cancún,
    # Mexico" → "Mexico City to Cancún" glued onto every stop's search
    # regardless of which leg it's on). Can't import that helper directly
    # (places.py already imports FROM this module — importing it back
    # here would be circular), so the same narrow fix is duplicated
    # inline: fall back to the country name (after the last comma) for a
    # multi-leg destination, otherwise keep the original single-city
    # behavior.
    if destination and (re.search(r"\bto\b", destination, re.IGNORECASE) or "(" in destination) and "," in destination:
        city = destination.rsplit(",", 1)[-1].strip()
    else:
        city = destination.split(",")[0].strip() if destination else ""
    # Task 7: dates for the hotel links — see _hotel_checkin_checkout.
    # Computed once per route (not per stop): every hotel stop in the
    # same route shares the same trip dates.
    checkin, checkout = _hotel_checkin_checkout(data.get("duration", ""))

    for day in data.get("days", []):
        for stop in day.get("stops", []):
            category = stop.get("category")
            name = (stop.get("name") or "").strip()
            is_specific = stop.get("is_specific_name", True)

            if category == "hotel":
                query = f"{name} {city}".strip() if is_specific else city
                stop["booking_url"] = _booking_affiliate_url(query, checkin, checkout)
                stop["expedia_url"] = _expedia_affiliate_url(query, checkin, checkout)
            elif category == "food":
                query = f"{name} {city}".strip()
                stop["booking_url"] = _google_maps_search_url(query)
            elif category in ("activity", "sight"):
                # Nov 2026 (task 14): used to fall back to a generic
                # "{city} tours activities" search whenever is_specific
                # was False — throwing away the one thing that made the
                # search useful. Unlike a hotel's made-up name (which
                # could search for a property that doesn't exist),
                # `name` for a non-specific activity/sight IS a real
                # descriptive phrase the AI wrote from what it saw (e.g.
                # "Snorkelling with sea turtles, Oahu") — exactly the
                # kind of text GetYourGuide's own fuzzy search is built
                # to match against, so it's used here either way.
                query = f"{name} {city}".strip()
                stop["booking_url"] = _getyourguide_affiliate_url(query)

    return data

SYSTEM_PROMPT = """You are an expert travel itinerary extraction AI with strong visual recognition skills.

You will receive a series of evenly-spaced video frames from a travel TikTok or YouTube video.

Your job:
1. Read ALL on-screen text: subtitles, captions, restaurant signs, hotel names, street signs, menus
2. Recognise famous landmarks, beaches, and geographic features by appearance
3. Identify restaurant/hotel logos and branding visible in the frames
4. Infer location from architectural style, landscape, vegetation, and language of signs
5. Combine all evidence to build a structured day-by-day itinerary

Return ONLY valid JSON — no markdown, no code fences, no explanation.

If you cannot confidently identify a real, specific geographic destination
from the frames and comments (e.g. the video turns out to be mostly indoors
with no identifiable city/region — a home cooking video, a studio vlog,
generic lifestyle content that slipped past the earlier travel filter),
do NOT guess or invent a destination. Set "destination" to exactly the
string "UNKNOWN" instead, and leave the rest of the fields as empty/minimal
as the schema allows — the caller checks for this exact sentinel and will
show the visitor a "this doesn't look like a travel video" message instead
of a fabricated itinerary full of invented Unknown hotels and flights.

Schema:
{
  "destination": "City, Country" (or exactly "UNKNOWN" — see above),
  "duration": "X days",
  "summary": "2-3 sentence personal intro in a specific tone — see the 'summary' rule below, don't improvise the structure",
  "travel_tips": ["3-5 short, practical, destination-specific tips a first-time visitor genuinely needs — see guidance below"],
  "price_category": "€ | €€ | €€€",
  "tags": ["0-3 of: most_popular, luxury, budget_friendly, exotic, mountain, city, beach"],
  "car_rental_recommended": true,
  "car_rental_note": "Short reason, e.g. 'This itinerary covers several towns best explored by car.' — empty string if car_rental_recommended is false",
  "days": [
    {
      "day": 1,
      "label": "Short evocative label (e.g. Arrival & Old Town)",
      "stops": [
        {
          "name": "Exact location name as a tourist would search it",
          "category": "hotel|sight|food|activity|beach|village",
          "description": "One sentence of plain observation — what you can actually see/read in the frames about this stop (appearance, setting, atmosphere, what's visible on screen). NOT a claim about history, fame, rank ('most'/'best'), or anything you can't actually see. This is raw internal material — a later fact-grounded pass rewrites the user-facing description from this plus verified Google Places data, so keep it concrete and visual, not a polished pitch.",
          "tip": "Practical tip from the creator, max 12 words (empty string if none)",
          "is_specific_name": true,
          "property_type": "Hotel stops only — e.g. Boutique Hotel, Beach Resort, Guesthouse, Design Hotel (empty string for non-hotel stops)",
          "area_label": "Hotel stops only — the neighbourhood/area, e.g. Old Town, Beachfront (empty string if not identifiable or not a hotel)",
          "transfer_note": "How to get here relative to the hotel or previous stop, in general terms — e.g. 'Short walk from Old Town', 'Boat transfer needed', 'Car recommended'. Empty string if you can't confidently judge this."
        }
      ]
    }
  ]
}

"travel_tips" — 3-5 short, practical, destination-specific tips a
first-time visitor genuinely needs to know before they go — the kind of
thing a knowledgeable local friend would warn them about, not generic
travel-blog filler. Good angles: local payment norms ("cash is still
expected at most small vendors"), a specific transit rule that trips
people up ("tap OUT as well as in on the metro card or you're charged the
maximum fare"), a common scam or pickpocket area to watch for, a tipping
or etiquette norm that differs from what a Western traveler expects, a
practical booking/timing tip (opening hours, when a popular sight sells
out, best time to avoid crowds). Every tip must be REAL and genuinely true
for this specific destination — never invent one. Bad examples to avoid:
"bring comfortable shoes", "stay hydrated", "learn a few local phrases" —
these apply everywhere and say nothing destination-specific. Each tip is
one short sentence, max ~20 words. Return an empty array only if you
genuinely can't produce confident, specific tips for this destination.

"price_category" — your best estimate of the overall trip's price level based
on what's actually visible: budget hostel/guesthouse, street food, public
transport → "€". Mid-range hotel, casual sit-down restaurants → "€€".
Luxury resort/5-star hotel, fine dining, private tours/boats → "€€€".
Default to "€€" only if there's genuinely no visible signal either way.

"tags" — pick only tags that clearly and honestly fit; it's fine to return
an empty array if none clearly apply. Don't force a fit:
  most_popular    → an iconic, extremely well-known destination/route
  luxury          → high-end hotel/dining visible, or price_category is €€€
  budget_friendly → hostels, street food, or price_category is €
  exotic          → tropical, remote, or culturally distinct from Western Europe/US
  mountain        → mountains, hiking, ski, or alpine village setting
  city            → primarily an urban destination
  beach           → primarily a coastal/beach destination

"car_rental_recommended" — set true only when there's real signal for it:
  - the video itself shows/mentions driving, a rental car, or road-tripping, OR
  - the stops span multiple distinct towns/areas that aren't realistically
    walkable or connected by an obvious single public transit line
  Set false for single-city, walkable, or public-transit-friendly itineraries
  (most city breaks). When true, "car_rental_note" is one short, concrete
  reason — not a generic "cars are convenient" filler.

"property_type" and "area_label" (hotel stops only) — describe the
property's style and neighbourhood the way a travel writer would (e.g.
"Boutique Hotel", "Old Town"). NEVER include a numeric rating or review
score (e.g. "8.9", "4.5 stars") — you have no access to real guest review
data, and inventing a number would present fabricated data as if it were
a genuine score. Leave both fields as empty strings for non-hotel stops,
or if genuinely nothing about style/area is visible.

"transfer_note" — a short, general sense of how someone gets from the
previous stop (or the hotel, for the first stop of the day) to this one —
e.g. "Short walk from Old Town", "Boat transfer needed", "Car recommended,
~20 min". Keep it qualitative. Do NOT state a precise number of minutes
you can't actually know from the video — vague, honest phrasing beats
false precision. Leave it an empty string whenever you're not reasonably
confident.

"is_specific_name" must be:
  true  — "name" is a real, searchable property/place name (e.g. "Hotel
          Arts Barcelona", "Ars Magna Hotel", "Café 67")
  false — you could NOT confirm a specific name, so "name" is a generic
          description you wrote yourself (e.g. "Resort hotel, Hurghada",
          "Cliffside restaurant with blue umbrellas, Positano")
This flag is used to decide whether to show a direct booking link or a
"search this area" fallback — get it right rather than defaulting to true.

Category guide:
  hotel    → accommodation, guesthouse, resort, Airbnb
             IMPORTANT: Las Vegas casino-resorts (Venetian, Bellagio, MGM Grand,
             Caesars Palace, etc.) are ALWAYS "hotel" even though they are also
             tourist attractions. If a place has the word "Resort", "Hotel", or
             "Suites" in its name, classify as "hotel" regardless of visual appearance.
  sight    → landmark, viewpoint, museum, cathedral, village, natural feature
  food     → restaurant, café, bar, market, beach club with food/drinks
  activity → water sport, hike, tour, boat trip, zip-line
  beach    → beach with no primary food/bar focus

Before giving up on a hotel/restaurant name, check every available clue:
  - Signage, awnings, matchbooks, receipts, chalkboards, menus, coasters
  - Staff uniform badges/logos, keycards, room-key fobs, welcome folders
  - Branded towels, robes, slippers, toiletries, minibar items
  - Pool furniture branding, umbrella logos, branded floats
  - Location tags or captions burned into the video itself
  - On-screen text overlays the creator added (hotel/restaurant names are
    often typed as captions even when not visible in the shot)
Only fall back to a generic description ("is_specific_name": false) after
genuinely checking for these — don't default to giving up early.

Rules:
- Always write "summary": 2-3 sentences, no more, in this specific
  personal tone (confirmed style, don't improvise a different structure):
    1. Open with a short, genuine compliment on the destination choice
       itself — "Nice instinct choosing X" / "Great pick" / "Smart call"
       style. Vary the exact phrasing between routes; don't reuse the same
       opener every time.
    2. A sentence built on real contrast or dynamic FROM THIS ROUTE'S
       ACTUAL STOPS, not a generic destination description — something
       like "X by day, Y by the end of the trip", grounded in what this
       specific itinerary actually contains (e.g. city sightseeing early
       on, something wilder or different later), never invented.
    3. Close with one sentence characterizing what KIND of traveler this
       specific route suits — "built for someone who wants…" — not a
       recap of what's included.
  Address the traveler indirectly, not literal "you" throughout (the
  opening compliment already implies it — see the example). Never use
  travel-brochure clichés ("has beautiful beaches", "rich culture",
  "vibrant city"). Example, for an Edinburgh route whose stops are Old
  Town sights early on and Arthur's Seat + a Highlands day trip later:
    "Nice instinct choosing Edinburgh — medieval streets by day, volcanic
    hills and Highlands by the end of the trip. This itinerary is built
    for someone who wants a city that still feels a little wild."
  This REPLACES the old "what makes the destination special, then what
  the route covers" structure — don't blend the two. Plain, engaging
  prose either way — not a bullet list, not marketing hype.
- Extract EVERY named location — do not summarise or skip stops
- If a place has no visible name after checking the clues above, describe it
  precisely and set "is_specific_name": false — e.g. "Cliffside restaurant
  with blue umbrellas, Positano"
- Group by day using this priority:
  1. If the video itself states day-by-day structure (e.g. "Day 1", "Day 2"
     captions or narration), use that structure exactly.
  2. Otherwise (e.g. a "Top 10 hidden gems in X" or "best places to visit"
     compilation with no day structure) — reason about how long each place
     actually takes to visit, using your knowledge of the destination, then
     group accordingly:
       - A major full-day attraction (theme park, day-trip island, large
         museum complex, multi-hour hike, safari, ski resort) gets its OWN
         day — don't pack anything else alongside it.
       - Quick stops (viewpoint, small church, photo spot, café, short
         walk) can be combined — several in one day IF they're also close
         to each other geographically.
       - A half-day attraction (large landmark, big market, boat tour)
         pairs with at most one or two quick stops, not several.
     There's no fixed number of stops per day — a day can have 1 stop if
     that's what its size warrants, or 4-5 if they're all quick and nearby.
     Label each day after its area or theme (e.g. "Old Town & Harbour",
     "West Coast Beaches") rather than leaving it generic.
  3. Only fall back to putting everything in Day 1 if there are 3 or fewer
     stops total, or if you cannot confidently place them geographically
     or judge their typical visit length.
- If a hotel/accommodation is identified, put it in Day 1 (it's the base
  the traveler returns to) even if the video mentions it later
- If you see fewer than 3 identifiable locations, mention it in Day 1 label
- NEVER invent places you cannot see or read in the frames
- Return only the JSON object"""


def dedupe_landmark_stops(itinerary: Itinerary) -> int:
    """
    Task 14 route-quality pass (Nov 2026): drops a "sight"-category stop
    that's a repeat of an earlier one already kept elsewhere in the SAME
    route. Confirmed live as a real extraction bug, not a hypothetical:
    "Palacio de Bellas Artes" showed up on BOTH Day 1 and Day 7 of a
    7-day Mexico City route — the AI reading the same landmark out of two
    different segments of a long video as if they were two different
    stops.

    Deliberately narrow — ONLY "sight". Hotel and food stops are NEVER
    touched here: a hotel legitimately repeats across every night of a
    multi-day stay, and a restaurant can legitimately be shown again as a
    return visit — confirmed two more real (correct, not-a-bug) cases in
    the same audit: a Milos route's hotel and a favourite restaurant both
    genuinely appearing on Day 1 AND Day 4. Only "sight" is unambiguous
    enough (a traveler doesn't typically re-visit the same landmark on
    unrelated days of the same trip) to safely auto-fix without a human
    call — every other category is left exactly as extracted.

    Matches on a normalized name (lowercased, punctuation stripped) in
    case the AI phrased the repeat slightly differently the second time.
    Mutates itinerary.days in place and returns how many stops were
    dropped (0 = nothing to do).
    """
    seen: set[str] = set()
    dropped = 0
    for day in itinerary.days:
        kept = []
        for stop in day.stops:
            if stop.category == "sight":
                key = re.sub(r"[^\w\s]", "", stop.name.lower()).strip()
                if key and key in seen:
                    print(f"[Dedup] Dropping duplicate sight stop '{stop.name}' (day {day.day})")
                    dropped += 1
                    continue
                if key:
                    seen.add(key)
            kept.append(stop)
        day.stops = kept
    return dropped


def analyse_frames(frame_paths: list[str], comments: list[dict] | None = None, caption: str = "") -> tuple[Itinerary, str, list[str], float]:
    """
    Sends all frames to Claude Sonnet and parses the JSON itinerary response.
    Raises ValueError if the response cannot be parsed.

    `comments` (optional): the video's real top comments (from
    extractor.fetch_top_comments), passed in as an identification aid —
    viewers frequently ask "what hotel is this??" and someone (often the
    creator) answers with the actual name, which the AI would otherwise
    never see since it only looks at frames. Only the comments themselves
    decide this — the AI is told explicitly not to guess a name it can't
    verify from either frames or comments.

    `caption` (optional, Nov 2026 — item 7, "missed stops"): the video's
    own title/description text (meta["title"] + meta["description"] from
    extractor.fetch_metadata/fetch_instagram_post), passed in for the
    same reason as comments — the AI currently only sees sampled frames,
    so a place mentioned by name in the caption but shown only briefly
    (or in a frame that fell between samples) is otherwise invisible to
    it. Same guardrail as comments: the caption only helps NAME/CONFIRM
    a place that's genuinely shown or implied — it's never license to
    add a stop that isn't actually in the video, and never raises
    is_specific_name on its own.

    Returns (itinerary, price_category, tags, cost_usd) — price_category
    and tags aren't part of the Itinerary content model (they're
    admin-curation fields stored separately via database.set_route_meta),
    so they're popped out of the raw response here rather than silently
    dropped. cost_usd is the real $ cost of this specific API call,
    computed from Anthropic's own token usage numbers in the response —
    not an estimate.
    """
    intro_text = (
        f"Here are {len(frame_paths)} evenly-spaced frames from a travel video. "
        "Please identify every travel location and return the structured JSON itinerary."
    )

    caption_text = (caption or "").strip()
    if caption_text:
        intro_text += (
            "\n\nHere is the video's own title/caption text, exactly as "
            "posted. It can help you NAME or CONFIRM a place that's shown "
            "or implied in the frames, and can point you to a place the "
            "frames only glimpse briefly. Do NOT add a stop that isn't "
            "actually shown in the video just because it's mentioned here, "
            "and do not treat a caption mention alone as confirmation "
            "strong enough to set is_specific_name: true — that still "
            "requires the name to be legible on-screen or confirmed in "
            "the comments.\n\n"
            f"{caption_text[:1500]}"
        )

    # Comments are genuinely useful for exactly one thing: naming a place
    # the frames alone don't confirm. Keep it short (top 10 by likes) and
    # be explicit that this is the only extra source of truth allowed —
    # nothing here should lower the bar for is_specific_name.
    if comments:
        top = sorted(comments, key=lambda c: c.get("likes", 0), reverse=True)[:10]
        comment_lines = "\n".join(f'- "{c.get("text", "").strip()}"' for c in top if c.get("text", "").strip())
        if comment_lines:
            intro_text += (
                "\n\nHere are this video's top comments (most-liked first). "
                "Some viewers ask what a specific hotel/restaurant/place is "
                "called, and sometimes the creator or another viewer answers "
                "with the real name — if so, use that confirmed name and set "
                "is_specific_name: true. Do NOT use comments to guess a name "
                "that isn't actually stated in them; if nothing here confirms "
                "a name, treat it exactly as if there were no comments at all.\n\n"
                f"{comment_lines}"
            )

    # Build multimodal content: intro text + all JPEG frames
    content: list[dict] = [{"type": "text", "text": intro_text}]

    for path in frame_paths:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            },
        })

    response = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    if response.stop_reason == "max_tokens":
        raise ValueError(
            "Claude's response was cut off before finishing (too many stops "
            "for the token limit). Try again — if this keeps happening, "
            "the itinerary may need to be split or max_tokens raised further."
        )

    raw = response.content[0].text.strip()

    # Strip accidental markdown code fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Claude's response wasn't valid JSON: {e}") from e

    data = _attach_booking_urls(data)

    price_category = data.pop("price_category", "") or "€€"
    tags = data.pop("tags", None) or []

    usage = getattr(response, "usage", None)
    cost_usd = 0.0
    if usage:
        cost_usd = (
            usage.input_tokens * _SONNET_INPUT_PER_MTOK
            + usage.output_tokens * _SONNET_OUTPUT_PER_MTOK
        ) / 1_000_000

    data["travel_tips"] = [str(t).strip() for t in (data.get("travel_tips") or []) if str(t).strip()]

    return Itinerary(**data), price_category, tags, cost_usd


_GROUNDED_DESCRIPTION_SYSTEM = """You write ONE short, USEFUL description (1-2 sentences, max ~30 words) for each travel-itinerary stop you're given, for a travel app's route page. The description should tell a traveler something they can actually use — what to do there, what it feels like, when to go, a quick practical tip — not just restate the address or category they can already see printed on the card next to it.

You will receive a JSON list of stops. For EACH stop you get:
- "name": the place's name
- "video_notes": raw, unpolished observations from a travel video showing this stop — may be empty string if there's no video (e.g. the traveler built this trip themselves rather than pasting a video link)
- "place_type", "address", "area_label", "rating", "user_rating_count", "price_level", "opening_hours_weekly": verified facts about the place from Google Places — any of these may be empty/zero if Google doesn't have that data for this place

ALLOWED — lean on this, it's what makes the description worth reading:
- What a traveler can actually do there (browse, stroll, eat, photograph, relax, people-watch)
- Atmosphere, in general terms (lively, quiet, scenic, cozy, romantic)
- General advice on when it's good to go (evening, sunset, a weekday morning) — ordinary travel-guide framing, not a specific claimed fact about this place that you weren't given
- A short, practical tip (what to pair it with, how to approach it, what to bring)
- Anything actually shown or said in video_notes

FORBIDDEN — these are the whole point of this task, never break them:
1. Never invent a checkable fact you weren't given — about THIS place's history, size, age, records, or anything else specific and verifiable. Generic travel-guide advice (see ALLOWED) is fine precisely because it isn't a checkable claim about this particular place — it reads the same whether or not Google gave you much to go on.
2. NO history, no years, no founding dates, no "built in", no "since".
3. NO invented numbers — don't state a size, distance, count, or age that wasn't given to you. (rating/user_rating_count/price_level ARE given to you, but don't restate them as prose — the app displays those separately; use them only to calibrate tone, e.g. a high rating can justify "popular" but never a specific invented number.)
4. NO superlative claims ("the best", "the most", "the only", "the biggest", "iconic", "legendary", "world-famous") unless that exact claim is explicitly present in video_notes — a verified Places rating is NOT license to call something "the best."
5. NO comparisons to other places, other cities, or "unlike anywhere else" framing, unless explicitly present in video_notes.
6. Only when you genuinely have nothing to hang advice on (no useful place_type, no video_notes, nothing) fall back to the plainest possible factual sentence from what little you do have — that should be the rare exception, not the default. A flat "A main street in central Bari" when you had "place_type: Street" to work with is a worse answer than reaching for ordinary travel-guide framing instead — e.g. "Bari's main shopping street, good for an evening stroll after dinner."
7. Never mention that you were given "data", "facts", or "video" — write a normal, natural sentence a traveler would read, just one that happens to be fully grounded.

Reply with ONLY valid JSON, no markdown fences:
{"descriptions": {"<exact stop name as given>": "description text", ...}}
Every stop name you were given must appear as a key, even if its description ends up being short and plain."""


def generate_grounded_descriptions(stops: list[dict]) -> tuple[dict[str, str], float]:
    """
    Task 8: ONE batched Haiku call per route (not per stop, for cost
    efficiency — same batching pattern the old generate_fun_facts used)
    that rewrites every stop's user-facing description so it's grounded
    ONLY in Google Places facts + the AI's own raw video observations —
    never history/years/invented numbers/superlatives/comparisons. Called
    AFTER places.enrich_itinerary_with_photos() has populated each stop's
    Places facts, for both video-extracted routes (video_notes = that
    stop's original AI-written `description`, i.e. what analyse_frames'
    SYSTEM_PROMPT calls "raw internal material") and Build Your Own Trip
    (no video at all, so video_notes="" for every stop — descriptions come
    purely from Places facts there).

    Each item in `stops` is shaped like:
      {"name", "video_notes", "place_type", "address", "area_label",
       "rating", "user_rating_count", "price_level", "opening_hours_weekly"}

    Returns ({stop_name: grounded_description}, cost_usd). Never raises —
    returns ({}, 0.0) on any failure (rate limit, bad response, network
    error); callers should leave stop.description unchanged (keep
    whatever it already was) for any stop missing from the result dict,
    rather than blanking it out.
    """
    if not stops:
        return {}, 0.0
    try:
        payload = [
            {
                "name": s.get("name", ""),
                "video_notes": s.get("video_notes", "") or "",
                "place_type": s.get("place_type", "") or "",
                "address": s.get("address", "") or "",
                "area_label": s.get("area_label", "") or "",
                "rating": s.get("rating", 0) or 0,
                "user_rating_count": s.get("user_rating_count", 0) or 0,
                "price_level": s.get("price_level", "") or "",
                "opening_hours_weekly": s.get("opening_hours_weekly", []) or [],
            }
            for s in stops
        ]
        response = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4000,
            system=_GROUNDED_DESCRIPTION_SYSTEM,
            messages=[{"role": "user", "content": json.dumps({"stops": payload})}],
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
        descriptions = {
            str(name): str(desc).strip()
            for name, desc in (result.get("descriptions") or {}).items()
            if str(desc).strip()
        }
        return descriptions, cost_usd
    except Exception as e:
        print(f"[GroundedDescriptions] generate_grounded_descriptions failed: {e}")
        return {}, 0.0


_TRIP_SUMMARY_SYSTEM = """You write a short intro for a travel app's route page — the same "summary" a video-extracted route gets (see SYSTEM_PROMPT's "summary" rule in ai_analyzer.py — keep this in sync with that one), just for a traveler who built their own trip instead of pasting a video link.

You'll receive a destination and the list of specific attractions/activities the traveler picked for their trip.

Write 2-3 sentences, no more, in this specific personal tone (confirmed style, don't improvise a different structure):
  1. Open with a short, genuine compliment on the destination choice itself — "Nice instinct choosing X" / "Great pick" / "Smart call" style. Vary the exact phrasing between routes; don't reuse the same opener every time.
  2. A sentence built on real contrast or dynamic FROM THE TRAVELER'S ACTUAL PICKS, not a generic destination description — something like "X by day, Y by the end of the trip", grounded in what they actually selected, never invented.
  3. Close with one sentence characterizing what KIND of traveler this specific selection suits — "built for someone who wants…" — not a recap of what's included.
Address the traveler indirectly, not literal "you" throughout (the opening compliment already implies it — see the example). Never use travel-brochure clichés ("has beautiful beaches", "rich culture", "vibrant city"). Example, for an Edinburgh trip whose picks are Old Town sights plus Arthur's Seat and a Highlands day trip:
  "Nice instinct choosing Edinburgh — medieval streets by day, volcanic hills and Highlands by the end of the trip. This itinerary is built for someone who wants a city that still feels a little wild."

Reply with ONLY the summary text, nothing else — no markdown, no surrounding quotes."""


def generate_trip_summary(destination: str, stop_names: list[str]) -> tuple[str, float]:
    """
    Cheap, standalone Haiku call that writes the same kind of 2-3 sentence
    destination + route summary a video-extracted route gets for free from
    the main Sonnet call (see SYSTEM_PROMPT's "summary" field) — Build Your
    Own Trip has no video to extract that from, so this fills the same gap
    generate_grounded_descriptions/generate_travel_tips fill for their own fields.

    Returns (summary, cost_usd). Never raises — returns ("", 0.0) on any
    failure, same non-fatal contract as generate_grounded_descriptions.
    """
    try:
        response = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=_TRIP_SUMMARY_SYSTEM,
            messages=[{"role": "user", "content": json.dumps({
                "destination": destination,
                "stops": stop_names,
            }, ensure_ascii=False)}],
        )
        usage = getattr(response, "usage", None)
        cost_usd = 0.0
        if usage:
            cost_usd = (
                usage.input_tokens * _HAIKU_INPUT_PER_MTOK
                + usage.output_tokens * _HAIKU_OUTPUT_PER_MTOK
            ) / 1_000_000
        summary = response.content[0].text.strip().strip('"').strip()
        return summary, cost_usd
    except Exception as e:
        print(f"[TripSummary] generate_trip_summary failed for '{destination}': {e}")
        return "", 0.0


_TRAVEL_TIPS_SYSTEM = """You write 3-5 short, practical, destination-specific travel tips for a travel app's "Tips & Tricks" section — the kind of thing a knowledgeable local friend would warn a first-time visitor about, not generic travel-blog filler.

Good angles: local payment norms ("cash is still expected at most small vendors outside the centre"), a specific transit rule that trips people up ("tap OUT as well as in on the metro card or you're charged the maximum fare"), a common scam or pickpocket area to watch for, a tipping or etiquette norm that differs from what a Western traveler expects, a practical booking/timing tip (opening hours, when a popular sight sells out, best time to avoid crowds).

Rules:
- Every tip must be REAL and genuinely true for this specific destination — never invent one. If you aren't confident enough in a specific detail, leave it out rather than guess.
- Bad examples to avoid — never write these or anything like them: "bring comfortable shoes", "stay hydrated", "learn a few local phrases", "respect the local culture". These apply everywhere and say nothing destination-specific.
- Each tip is one short sentence, max ~20 words.
- Return between 3 and 5 tips. Fewer than 3 only if you genuinely can't produce that many confident, specific tips for this destination.

Reply with ONLY valid JSON, no markdown fences:
{"tips": ["tip one", "tip two", "tip three"]}"""


def generate_travel_tips(destination: str) -> tuple[list[str], float]:
    """
    Cheap, standalone Haiku call that returns 3-5 real, practical,
    destination-specific travel tips for `destination` — used by Build
    Your Own Trip (which has no Sonnet video-analysis call to get tips for
    free from) and to backfill routes generated before this field existed,
    same role generate_grounded_descriptions plays for stop descriptions. New video-extracted
    routes get travel_tips for free as part of the same analyse_frames()
    call instead (see SYSTEM_PROMPT above).

    Returns (tips, cost_usd). Never raises — returns ([], 0.0) on any
    failure, same non-fatal contract as generate_grounded_descriptions.
    """
    try:
        response = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=_TRAVEL_TIPS_SYSTEM,
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
        tips = [str(t).strip() for t in (result.get("tips") or []) if str(t).strip()]
        return tips, cost_usd
    except Exception as e:
        print(f"[TravelTips] generate_travel_tips failed for '{destination}': {e}")
        return [], 0.0


_CALENDAR_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Ordered so iteration is deterministic (a plain set's iteration order
# isn't guaranteed) — the single source of truth for both the region-by-
# region generation loop below and _VALID_REGIONS (still referenced by
# models.py's MonthDestination docstring and main.py).
_REGION_ORDER = ["Europe", "Asia", "Africa", "North America", "South America", "Oceania"]
_VALID_REGIONS = set(_REGION_ORDER)

# Real bug this replaces (confirmed live 2026-09-12): the previous design
# was ONE call for the whole year, ALL regions mixed together, under a
# strict "every destination unique across all 12 months" rule — at
# target=10/month that demanded ~120 genuinely distinct, confidently-
# well-known countries out of ~195 that exist. The model burned through
# its "safe, obviously right" answers on the early months and later
# months (especially Sep-Dec, iterated last in the old dedup pass) fell
# back to the emergency floor: live counts were Jan-May 10/10 each, then
# a steady decline to Oct/Nov/Dec at 3 (the floor) — not a filter bug
# (regions were correctly tagged throughout), a genuine supply/demand
# mismatch between the uniqueness rule and the raised target.
#
# New design, per Gerry: one call PER REGION (6 total) instead of one
# call for everything, and destinations MAY repeat across months now —
# a place can be a great pick in more than one month for a genuinely
# different, specific reason (e.g. the Canary Islands as a November warm
# escape vs. an April shoulder-season pick), capped at 3 appearances/year
# so one "safe, pleasant year-round" country can't become generic filler.
# This also means "region" is no longer something the model has to get
# right — it's simply which loop iteration produced the entry, so it can
# never come back malformed/missing (no more validating it against
# _VALID_REGIONS at all).
# v11 (Nov 2026, Gerry's decision after the v10 audit): v10 kept this
# list as PROSE and trusted the model to (a) only cite an event from it,
# in its right month, for its right location, and (b) never mention a
# natural phenomenon at all. Both trusts were confirmed broken on a real
# v10 generation — not edge cases, 50+ entries: Gion Matsuri cited in
# June while the model's own text admitted "begins in July"; Obon cited
# in July when it's only ever mid-August; "Post-Mardi Gras"/"Post-Timkat"
# used as a month anchor outside the event's own month; two invented
# events not on the list at all ("Brisbane Festival", "Auckland
# Anniversary Day"); and 51 explicit mentions of banned natural
# phenomena (aurora, wildebeest calving/migration, bloom/blossom,
# autumn foliage, coral spawning, whale season, migratory birds) despite
# the prompt saying "NO natural phenomenon, ever". Per Gerry: a prompt
# instruction is no longer enough on its own — this is now STRUCTURED
# data the CODE validates after generation (see generate_month_calendar
# below), not prose the model has to interpret and self-police. The
# model only ever supplies an `event_id` from this table; the displayed
# event name and the month/location check both come from here, in code,
# never from the model's own text.
#   id -> {"name": display name shown to users,
#          "months": set of 1-12 ints this event is valid for,
#          "locations": None (worldwide/no restriction) or a list of
#          lowercase substrings — the destination name must contain at
#          least one for this event_id to be accepted for it}
# Changes from the v10 prose list, per Gerry's review of it:
#   - Glastonbury Festival REMOVED entirely — it has fallow/no-festival
#     years (including 2026), so "late Jun" isn't reliably true every
#     year the way the other fixed/predictable events are.
#   - Vesak widened to Apr-Jun (was Apr-May) — Thailand's Vesak fell in
#     June in 2023; the lunar date genuinely moves enough that Apr-May
#     alone is too narrow for Thailand specifically.
#   - Carnival / Mardi Gras locations now explicitly include Salvador
#     and Recife, Brazil (two of Brazil's biggest, most famous Carnival
#     cities — the v10 list only named Rio, so otherwise-correct Feb-Mar
#     Carnival picks for Salvador/Recife had nowhere valid to point to).
_VERIFIED_EVENTS: dict[str, dict] = {
    "new_years_day": {"name": "New Year's Day", "months": {1}, "locations": None},
    "cape_town_carnival": {"name": "Cape Town Minstrel Carnival / Kaapse Klopse", "months": {1}, "locations": ["cape town"]},
    "voodoo_festival": {"name": "Vodoun Day / Voodoo Festival", "months": {1}, "locations": ["ouidah", "benin"]},
    "timkat": {"name": "Timkat (Ethiopian Epiphany)", "months": {1}, "locations": ["ethiopia"]},
    "australia_day": {"name": "Australia Day", "months": {1}, "locations": ["australia"]},
    "harbin_ice_festival": {"name": "Harbin Ice and Snow Festival", "months": {1, 2}, "locations": ["harbin"]},
    "lunar_new_year": {"name": "Lunar New Year (Chinese New Year / Tet / Seollal)", "months": {1, 2}, "locations": ["china", "vietnam", "korea", "hong kong", "taiwan", "singapore", "east asia"]},
    "waitangi_day": {"name": "Waitangi Day", "months": {2}, "locations": ["new zealand"]},
    "carnival_mardi_gras": {"name": "Carnival / Mardi Gras", "months": {2, 3}, "locations": ["rio de janeiro", "rio", "venice", "new orleans", "oruro", "sydney", "salvador", "recife"]},
    "holi": {"name": "Holi", "months": {2, 3}, "locations": ["india"]},
    "st_patricks_day": {"name": "St. Patrick's Day", "months": {3}, "locations": ["ireland"]},
    "nowruz": {"name": "Nowruz (Persian New Year)", "months": {3}, "locations": ["iran", "uzbekistan", "tajikistan", "kazakhstan", "kyrgyzstan", "azerbaijan", "afghanistan"]},
    "vendimia": {"name": "Fiesta Nacional de la Vendimia (grape harvest festival)", "months": {2, 3}, "locations": ["mendoza"]},
    "sxsw": {"name": "South by Southwest (SXSW)", "months": {3}, "locations": ["austin"]},
    "easter": {"name": "Easter (incl. Semana Santa)", "months": {3, 4}, "locations": None},
    "passover": {"name": "Passover", "months": {3, 4}, "locations": ["israel"]},
    "songkran": {"name": "Songkran (Thai New Year)", "months": {4}, "locations": ["thailand"]},
    "qingming": {"name": "Qingming Festival (Tomb-Sweeping Day)", "months": {4}, "locations": ["china"]},
    "anzac_day": {"name": "ANZAC Day", "months": {4}, "locations": ["australia", "new zealand"]},
    "coachella": {"name": "Coachella", "months": {4}, "locations": ["california"]},
    "vesak": {"name": "Vesak (Buddha's Birthday)", "months": {4, 5, 6}, "locations": ["thailand", "sri lanka"]},
    "cinco_de_mayo": {"name": "Cinco de Mayo", "months": {5}, "locations": ["mexico"]},
    "cannes_film_festival": {"name": "Cannes Film Festival", "months": {5}, "locations": ["cannes", "france"]},
    "kentucky_derby": {"name": "Kentucky Derby", "months": {5}, "locations": ["kentucky"]},
    "dragon_boat_festival": {"name": "Dragon Boat Festival", "months": {5, 6}, "locations": ["china", "east asia", "hong kong", "taiwan"]},
    "vivid_sydney": {"name": "Vivid Sydney", "months": {5, 6}, "locations": ["sydney"]},
    "inti_raymi": {"name": "Inti Raymi (Festival of the Sun)", "months": {6}, "locations": ["cusco"]},
    "wimbledon": {"name": "Wimbledon", "months": {6, 7}, "locations": ["england", "london"]},
    "canada_day": {"name": "Canada Day", "months": {7}, "locations": ["canada"]},
    "independence_day_usa": {"name": "Independence Day", "months": {7}, "locations": ["usa", "united states"]},
    "san_fermin": {"name": "San Fermín (Running of the Bulls)", "months": {7}, "locations": ["pamplona"]},
    "bastille_day": {"name": "Bastille Day", "months": {7}, "locations": ["france"]},
    "gion_matsuri": {"name": "Gion Matsuri", "months": {7}, "locations": ["kyoto"]},
    "tour_de_france": {"name": "Tour de France", "months": {7}, "locations": ["france"]},
    "naadam": {"name": "Naadam Festival", "months": {7}, "locations": ["mongolia"]},
    "calgary_stampede": {"name": "Calgary Stampede", "months": {7}, "locations": ["calgary"]},
    "durban_july": {"name": "Durban July (horse race)", "months": {7}, "locations": ["durban"]},
    "edinburgh_fringe": {"name": "Edinburgh Festival Fringe", "months": {8}, "locations": ["edinburgh", "scotland"]},
    "obon": {"name": "Obon", "months": {8}, "locations": ["japan"]},
    "notting_hill_carnival": {"name": "Notting Hill Carnival", "months": {8}, "locations": ["london"]},
    "la_tomatina": {"name": "La Tomatina", "months": {8}, "locations": ["buñol", "bunol", "valencia"]},
    "venice_film_festival": {"name": "Venice Film Festival", "months": {8, 9}, "locations": ["venice", "italy"]},
    "oktoberfest": {"name": "Oktoberfest", "months": {9, 10}, "locations": ["munich", "bavaria"]},
    "mid_autumn_festival": {"name": "Mid-Autumn Festival", "months": {9, 10}, "locations": ["china", "east asia", "hong kong", "taiwan", "vietnam", "korea"]},
    "golden_week": {"name": "Golden Week (National Day)", "months": {10}, "locations": ["china"]},
    "balloon_fiesta": {"name": "Albuquerque International Balloon Fiesta", "months": {10}, "locations": ["albuquerque"]},
    "halloween": {"name": "Halloween", "months": {10}, "locations": None},
    "diwali": {"name": "Diwali", "months": {10, 11}, "locations": ["india", "nepal"]},
    "day_of_the_dead": {"name": "Día de los Muertos (Day of the Dead)", "months": {11}, "locations": ["mexico"]},
    "melbourne_cup": {"name": "Melbourne Cup", "months": {11}, "locations": ["melbourne"]},
    "thanksgiving": {"name": "Thanksgiving", "months": {11}, "locations": ["usa", "united states"]},
    "christmas_markets": {"name": "European Christmas Markets", "months": {11, 12}, "locations": ["germany", "austria", "france", "czech", "poland", "uk", "united kingdom", "vienna", "strasbourg", "prague", "kraków", "krakow", "warsaw"]},
    "fiestas_de_quito": {"name": "Fiestas de Quito", "months": {12}, "locations": ["quito"]},
    "hanukkah": {"name": "Hanukkah", "months": {11, 12}, "locations": ["israel"]},
    "christmas": {"name": "Christmas", "months": {12}, "locations": None},
    "boxing_day": {"name": "Boxing Day", "months": {12}, "locations": ["uk", "united kingdom", "canada", "australia", "new zealand"]},
    "new_years_eve": {"name": "New Year's Eve", "months": {12}, "locations": None},
}


def _events_table_text() -> str:
    """Compact `id | name | months | where` table for the prompt — see
    _VERIFIED_EVENTS's comment for why this is structured data the model
    only ever references by id, not prose it has to interpret."""
    lines = []
    for eid, ev in _VERIFIED_EVENTS.items():
        months_txt = ",".join(_CALENDAR_MONTHS[m - 1][:3] for m in sorted(ev["months"]))
        where_txt = ", ".join(ev["locations"]) if ev["locations"] else "worldwide"
        lines.append(f"{eid} | {ev['name']} | {months_txt} | {where_txt}")
    return "\n".join(lines)


# v11 (Nov 2026): the v10 prompt's categorical "NO natural phenomenon,
# ever" instruction was confirmed broken on a real generation — 51
# entries still named one (aurora/northern lights, wildebeest calving/
# migration, bloom/blossom, autumn foliage/fall colors, coral spawning,
# whale season, migratory birds). Per Gerry: this can no longer be a
# prompt-only rule — it's now a CODE-level filter applied to every
# entry's reason_detail (and destination name) after generation,
# regardless of what the prompt says, with explicit synonyms for each
# phenomenon category rather than one keyword each (e.g. "fall colors"
# as well as "foliage", "leaves turning" as well as "autumn colors") —
# the model found ways to describe these phenomena without using the
# single keyword a narrower filter would have caught.
_BANNED_PHENOMENON_RE = re.compile(
    r"\bmigrat\w*|\bbloom\w*|\bblossom\w*|\baurora\b|\bnorthern lights\b|"
    r"\bwhale\w*|\bcalv(e|ing)\b|\bspawn\w*|\bnest(ing)?\b|\brut\b|"
    r"\bfoliage\b|\beclipse\b|\bturtle\w*|\bhatch(ing)?\b|\bwildebeest\b|"
    r"\bfirefl(y|ies)\b|\bfall colo[u]?rs\b|\bautumn colo[u]?rs\b|"
    r"\bleaves (turn\w*|chang\w*)\b|\bsupermoon\b|\bmeteor shower\b|"
    r"\btulip\b|\bwildflower\w*|\bcherry blossom\w*|\bmagnolias?\b",
    re.I,
)

_REGION_CALENDAR_SYSTEM_TEMPLATE = """You are building a year-long "best places to visit this month" calendar for {region} ONLY, for GetWay, a travel app's homepage section.

For EACH of the 12 months (January through December), propose 6-8 real, well-known, genuinely popular tourist destinations (a country, region, or specific city) IN {region} — places with an actual, established tourism industry and real traveler demand, NEVER an event's own name, NEVER a generic phrase. A downstream step in CODE (not something you write, and not something you need to get right by following instructions alone) re-validates every one of these against Google Places and the EVENTS TABLE below and drops whatever doesn't check out, keeping only ~3-4 per month — so propose the full 6-8 even for less-obvious picks; a short list here just means fewer survive validation, never more.

For each destination, set reason_type to exactly one of "weather", "crowds", "prices", or "event":
- "weather" / "crowds" / "prices": write reason_detail as one short, specific clause (max ~16 words) — concrete to that exact month, never a generic "nice weather" that could apply any time of year. Leave event_id null.
- "event": set event_id to one id from the EVENTS TABLE below — ONLY when the current month you're writing is one of that event's listed months AND the destination is (or genuinely is part of) one of that event's listed locations. Never invent an event_id, never use one for the wrong month or a different location. reason_detail should describe what a visitor would see or do (max ~14 words) WITHOUT repeating the event's name or date — the app inserts the real name and timing automatically from the table, your text is just color.

ABSOLUTE RULES for reason_detail, every reason_type, no exceptions — a CODE-level filter checks these after you respond and silently drops any entry that breaks them, so there is no advantage to bending them:
1. Never name, hint at, or hedge toward any festival/holiday/carnival/celebration other than through a valid event_id as described above. Phrases like "festival season" or "local celebrations" with no valid event_id are OUT — use weather/crowds/prices instead.
2. Never mention ANY natural phenomenon, in any phrasing — no bloom/blossom/wildflowers, no autumn foliage/fall colors/leaves turning, no wildlife migration/calving/nesting/spawning, no aurora/northern lights, no whale season, nothing in this category, for any destination, ever. Use weather/crowds/prices instead.
3. The destination itself must always be a real, independently-bookable place — e.g. "Kyoto, Japan", never "Gion Matsuri (Kyoto, Japan)"; "Munich, Germany", never "Oktoberfest". If you can't name the place without naming the event, pick a different destination.

EVENTS TABLE — id | name | valid month(s) | where (usable ONLY as described above; never write the name/location text into reason_detail yourself):
{events_table}

The SAME destination MAY appear in more than one month, but ONLY with a genuinely different, specific reason each time — never the same or near-identical reason reused. No destination should appear more than 3 times across the whole year.

Across the FULL YEAR (all 12 months combined), {region} has dozens of real, well-known destinations beyond its 3-4 most famous names — cover a genuine geographic spread, don't default to the same handful reused with reworded reasons.

Reply with ONLY valid JSON, no markdown fences, all 12 month keys present:
{{"January": [{{"destination": "...", "reason_type": "weather", "event_id": null, "reason_detail": "..."}}, ...], "February": [...], "March": [...], "April": [...], "May": [...], "June": [...], "July": [...], "August": [...], "September": [...], "October": [...], "November": [...], "December": [...]}}"""

# v11 (Nov 2026): target lowered from 10 to 4 per Gerry's decision — with
# every entry now also validated against Google Places and the events
# table in code (see generate_month_calendar), a smaller, fully-verified
# set beats a larger one padded with whatever survives. No ABSOLUTE_FLOOR
# below this: if a region genuinely has fewer than 4 legitimate,
# validated picks for a given month, that's what gets shown.
TARGET_PER_REGION_MONTH = 4


def generate_month_calendar() -> tuple[dict[str, list[dict]], float]:
    """
    6 Haiku calls (one per region in _REGION_ORDER) that together build a
    full year's "best places to visit this month" calendar — homepage
    "Best places to visit this month" section, filterable by continent.

    One call per region rather than one call for the whole year: lets
    each region's picks compete only against themselves for the (now
    per-region, not global) per-month cap, and lets a destination
    legitimately reappear across different months (see
    _REGION_CALENDAR_SYSTEM_TEMPLATE) instead of every destination in the
    whole year needing to be unique — see this module's comment above
    _REGION_CALENDAR_SYSTEM_TEMPLATE for the real supply/demand problem
    that caused with the previous single-call, global-uniqueness design.

    A failure on one region's call is logged and that region simply ends
    up with no entries for the year, rather than losing the other 5
    regions too — never raises overall; returns ({}, 0.0) only if every
    single region's call fails.

    v11 (Nov 2026): the model's raw output per entry is now structured
    ("destination", "reason_type", "event_id", "reason_detail" — see
    _REGION_CALENDAR_SYSTEM_TEMPLATE) and every entry is independently
    re-validated here in CODE before it's kept, not trusted from the
    prompt alone (v10's prompt-only categorical bans were confirmed
    broken on a real generation — see _VERIFIED_EVENTS/
    _BANNED_PHENOMENON_RE's comments for specifics):
      1. Reject if reason_detail or destination matches
         _BANNED_PHENOMENON_RE (natural phenomena, any phrasing).
      2. If reason_type is "event": look up event_id in
         _VERIFIED_EVENTS and reject unless the current month and the
         destination's location both check out against that table —
         the DISPLAYED event name/text always comes from the table,
         never from the model's own words.
      3. Reject if `destination` doesn't resolve to a real place via
         Google Places (places.destination_is_real_place) — catches an
         event's own name being used as the "destination" field itself.
    Only entries that survive all three end up in raw_pools; the final
    per-region-per-month selection + year-wide 3x repeat cap below is
    unchanged from v10.

    Returns ({month_name: [{"name", "reason", "region"}, ...]}, total_cost_usd),
    entries from all 6 regions concatenated per month. "region" is set
    from which loop iteration produced the entry, never parsed from the
    model's own output — it can't come back missing or malformed.
    Photos are fetched separately per-destination at request time (see
    main.py's use of places._get_destination_gallery_unsplash) rather
    than baked in here, same as before.
    """
    calendar: dict[str, list[dict]] = {month: [] for month in _CALENDAR_MONTHS}
    total_cost_usd = 0.0
    any_succeeded = False
    # raw_pools[region][month] = every valid candidate for that region+month,
    # in the AI's own ranked order, NOT yet cut down to TARGET_PER_REGION_MONTH
    # or checked against the year-wide repeat cap — both of those happen
    # together in the final selection pass below, which needs to see past
    # rank 10 to be able to backfill a dropped slot with the next-ranked
    # candidate instead of just losing it.
    raw_pools: dict[str, dict[str, list[dict]]] = {}

    # Deferred import — places.py does `from ai_analyzer import ...` at
    # its own module level, so importing it back at ai_analyzer's module
    # level would be circular (see the duplicated-inline-fix comment
    # above in this file's affiliate-link code for the same constraint).
    # Safe here because by the time this function actually RUNS, both
    # modules have already finished loading.
    import places as _places_module

    # Local cache (this call only — a fresh one every regeneration, which
    # is fine since this only runs on an explicit manual admin trigger):
    # the same destination name can legitimately appear across several
    # months/regions in one run (e.g. a destination with 2-3 genuinely
    # different seasonal reasons), and this is a real network call
    # (Google Places), so avoid re-querying it every time.
    _place_check_cache: dict[str, bool] = {}

    def _destination_is_real_place_cached(name: str) -> bool:
        key = name.strip().lower()
        if key not in _place_check_cache:
            _place_check_cache[key] = _places_module.destination_is_real_place(name)
        return _place_check_cache[key]

    for region_idx, region in enumerate(_REGION_ORDER):
        # CONFIRMED root cause (2026-09-13), via a local run with real API
        # access and the temporary _last_generation_errors capture: every
        # region except Oceania was hitting max_tokens=4000 and getting cut
        # off mid-string, producing a JSONDecodeError ("Unterminated
        # string...") every single time, deterministically — not a
        # rate-limit or transient issue (retrying the identical
        # over-budget request just fails the same way twice). Oceania
        # alone happened to have few enough genuine destinations to fit
        # under 4000 tokens, which is why it looked like a "specific
        # regions fail" pattern rather than a size problem. Asking for up
        # to TARGET_PER_REGION_MONTH entries × 12 months in one region
        # call needs more headroom than the old single call did per
        # region (that one drew from one shared ~10-15/month pool across
        # ALL 6 regions combined). Bumped to 8000, then to 16000 once the
        # prompt started asking for 15/month instead of 10-12 (see below)
        # — that raised real output size enough (~9200 output tokens seen
        # live for Asia) that 8000 was cutting it close again. Proactive
        # inter-call spacing + retry are kept as a (harmless) safety net
        # for genuine transient errors, not the fix for this.
        if region_idx > 0:
            time.sleep(1.0)
        result = None
        for attempt in range(2):
            try:
                response = _client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=16000,
                    system=_REGION_CALENDAR_SYSTEM_TEMPLATE.format(region=region, events_table=_events_table_text()),
                    messages=[{"role": "user", "content": f"Generate the {region} 12-month calendar."}],
                )
                if response.stop_reason == "max_tokens":
                    raise ValueError(
                        f"{region} response was cut off at max_tokens — "
                        "output too long even at 16000 tokens."
                    )
                usage = getattr(response, "usage", None)
                if usage:
                    total_cost_usd += (
                        usage.input_tokens * _HAIKU_INPUT_PER_MTOK
                        + usage.output_tokens * _HAIKU_OUTPUT_PER_MTOK
                    ) / 1_000_000
                raw = response.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                result = json.loads(raw)
                break
            except Exception as e:
                is_last_attempt = attempt == 1
                print(f"[MonthCalendar] {region} call failed (attempt {attempt + 1}/2): {type(e).__name__}: {e}")
                if not is_last_attempt:
                    time.sleep(1.5)

        if result is None:
            continue
        any_succeeded = True

        try:
            region_pool: dict[str, list[dict]] = {}
            for month in _CALENDAR_MONTHS:
                midx = _CALENDAR_MONTHS.index(month) + 1
                entries = result.get(month) or []
                seen: set[tuple[str, str]] = set()
                kept = []
                for d in entries:
                    destination = str(d.get("destination") or "").strip()
                    reason_type = str(d.get("reason_type") or "").strip().lower()
                    event_id = d.get("event_id")
                    reason_detail = str(d.get("reason_detail") or "").strip()

                    if not destination or len(destination) > 48 or not reason_detail:
                        continue

                    # v11 code-level filter #1 (see _BANNED_PHENOMENON_RE's
                    # comment): natural-phenomenon ban is no longer a
                    # prompt instruction the model has to self-police —
                    # reject outright, regardless of reason_type, if either
                    # field matches, since v10 proved the model will still
                    # describe these without using an obvious keyword.
                    if _BANNED_PHENOMENON_RE.search(reason_detail) or _BANNED_PHENOMENON_RE.search(destination):
                        continue

                    if reason_type == "event":
                        # v11 code-level filter #2: the event's real name
                        # and whether it's even allowed this month/location
                        # is looked up here, in code, from _VERIFIED_EVENTS
                        # — never trusted from the model's own text. Any
                        # mismatch (unknown id, wrong month, wrong/missing
                        # location) drops the entry rather than trying to
                        # salvage it as a weather/crowds/prices pick, since
                        # reason_detail was written assuming the event
                        # context and may not stand alone sensibly.
                        ev = _VERIFIED_EVENTS.get(str(event_id or "").strip())
                        if not ev or midx not in ev["months"]:
                            continue
                        if ev["locations"] and not any(loc in destination.lower() for loc in ev["locations"]):
                            continue
                        final_reason = f"{ev['name']} — {reason_detail}"
                    elif reason_type in ("weather", "crowds", "prices"):
                        final_reason = reason_detail
                    else:
                        continue

                    # v11 code-level filter #3: the destination itself
                    # must resolve to a real, independently-bookable place
                    # — catches the "event name used as the destination"
                    # bug found in the v10 audit (e.g. "Thanksgiving, USA
                    # nationwide", "Dragon Boat Festival (China, East
                    # Asia)"), which the phenomenon/event checks above
                    # don't catch on their own since those names aren't
                    # phenomena and often don't even set reason_type to
                    # "event". See places.destination_is_real_place.
                    if not _destination_is_real_place_cached(destination):
                        continue

                    key = (destination.strip().lower(), final_reason.strip().lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    kept.append({"name": destination, "reason": final_reason, "region": region})
                region_pool[month] = kept
            raw_pools[region] = region_pool
        except Exception as e:
            print(f"[MonthCalendar] {region} call failed: {type(e).__name__}: {e}")
            continue

    if not any_succeeded:
        return {}, 0.0

    # Final selection + global repeat cap, combined (2026-09-13/14):
    # confirmed via a real live run that the system prompt's own "max
    # 3x/year" instruction is NOT reliably followed — Atacama Desert
    # showed up in all 12 months in one test, Galápagos in 10, Cartagena
    # in 8. A first attempt at fixing this trimmed each region+month to
    # TARGET_PER_REGION_MONTH *before* applying the cap, so a pick that
    # got dropped for hitting the cap just lost its slot outright — no
    # attempt to fill it from that region+month's own remaining
    # candidates. Confirmed live this visibly thinned out Africa/South
    # America in H2 months even though their raw pools still had
    # legitimate untapped candidates ranked just past position 10. Fixed
    # by walking raw_pools in rank order per (month, region), skipping —
    # not stopping at — any candidate whose name has already hit
    # MAX_REPEATS_PER_YEAR, so the next-ranked candidate can fill that
    # slot instead. Months are processed chronologically so a name's
    # occurrence count only ever reflects genuinely earlier months. No
    # floor if a region+month's pool runs out of eligible candidates
    # before reaching 10 — same "no artificial floor" design as
    # TARGET_PER_REGION_MONTH always had.
    MAX_REPEATS_PER_YEAR = 3
    name_occurrence_count: dict[str, int] = {}
    for month in _CALENDAR_MONTHS:
        for region, region_pool in raw_pools.items():
            selected = []
            for d in region_pool.get(month, []):
                key = d["name"].strip().lower()
                if name_occurrence_count.get(key, 0) >= MAX_REPEATS_PER_YEAR:
                    continue
                selected.append(d)
                name_occurrence_count[key] = name_occurrence_count.get(key, 0) + 1
                if len(selected) >= TARGET_PER_REGION_MONTH:
                    break
            calendar[month].extend(selected)

    return calendar, total_cost_usd


_VIBE_MATCH_SYSTEM = """You are matching a traveler's "find your travel vibe" quiz answers to the single real-world travel destination that best fits ALL of their answers combined, for GetWay, a travel itinerary app.

You will receive 7 question/answer pairs, in this order: who the trip is for (solo / partner / friends / family), terrain, WHY they're really going (their emotional driver — recharge / adventure / discover / connect), budget, the scene they want after dark, the one concrete thing they want more of (photos / food / culture / hidden gems), and which world region pulls them.

Your job:
1. Weigh ALL 7 answers together as one combined persona — never let a single answer override or ignore another. Treat "who's traveling" as close to a hard constraint: a family answer should never land on a party-hostel destination just because another answer said "nightlife" (read that as lively evenings, not a rave); a romantic-partner answer shouldn't land somewhere built around big group activities. The "why are you really going" answer is the strongest signal for WHICH KIND of place fits — "recharge" points toward calm/wellness-friendly destinations even if terrain says "city"; "adventure" points toward places with real outdoor/adrenaline options even if budget is modest. The "what do you want more of" answer should be genuinely reflected in the place you pick, not just its terrain or region — a traveler who wants "great food" needs a destination actually known for its food scene, not just any city in the right region. Every answer should be visible in why you picked the place, not just the loudest one.
2. Pick ONE real, specific, well-known travel destination (a city, town, or clearly-defined region — never a whole country) that a traveler could actually book and visit. Use real knowledge of the place: its actual price level, food/culture scene, terrain, nightlife, and vibe must genuinely fit the combined persona, not just share one keyword with one answer.
3. Respect their region answer (Europe / Asia / Americas / Africa & Middle East) — the destination must be in that region. If they answered "Anywhere" instead, you're free to pick the single best-fitting destination worldwide — don't default to any one region out of habit.
4. Write one short, engaging blurb in a travel-writer's voice (not a dry description) explaining why this place fits THEM — reference the specific combination of preferences that led you there, not a generic postcard line.

Reply with ONLY valid JSON, no markdown fences:
{"destination": "City, Country", "blurb": "One engaging sentence, max ~25 words."}"""


def generate_vibe_match(answers: list[dict]) -> tuple[dict, float]:
    """
    AI-backed match for the homepage "find your travel vibe" quiz
    (index.html) — reasons over ALL 7 quiz answers together rather than a
    tag-tally-plus-fixed-20-destination-list, so a real dining/budget
    preference can't be silently outvoted by whichever terrain answer got
    the most votes, and the result isn't capped at a small curated pool.

    Real bug this replaces: the old client-side matching tallied a single
    "top tag" (e.g. mountain) and only used budget/food as a tiebreaker
    among destinations already in that tag's pool — for regions where that
    pool had exactly one entry (e.g. Asia's only "mountain" destination was
    Everest Base Camp), the tiebreaker had nothing to break toward and a
    fine-dining answer got silently ignored. An AI call reasoning over all
    7 answers at once, against real-world knowledge instead of a fixed
    list, doesn't have that failure mode.

    `answers` is a list of {"question": ..., "answer": ...} dicts, in quiz
    order. Returns ({"destination": ..., "blurb": ...}, cost_usd), or
    ({}, 0.0) on any failure — never raises, same non-fatal philosophy as
    generate_grounded_descriptions, so the frontend can fall back to its
    own static-list matching rather than breaking the quiz.
    """
    try:
        response = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=_VIBE_MATCH_SYSTEM,
            messages=[{"role": "user", "content": json.dumps({"answers": answers}, ensure_ascii=False)}],
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
        destination = str(result.get("destination") or "").strip()
        blurb = str(result.get("blurb") or "").strip()
        if not destination:
            return {}, cost_usd
        return {"destination": destination, "blurb": blurb}, cost_usd
    except Exception as e:
        print(f"[VibeQuiz] generate_vibe_match failed: {type(e).__name__}: {e}")
        return {}, 0.0
