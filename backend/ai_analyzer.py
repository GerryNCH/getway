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
import urllib.parse

import anthropic
from models import Itinerary

_client = anthropic.Anthropic()

# Claude Sonnet 4.6 standard rate, $ per million tokens (verified July 2026 —
# update these two numbers if Anthropic changes pricing or analyse_frames'
# model string below is changed to a different model).
_SONNET_INPUT_PER_MTOK = 3.00
_SONNET_OUTPUT_PER_MTOK = 15.00

# Claude Haiku 4.5 standard rate, $ per million tokens — same figures used
# in quality_check.py and troll_filter.py. Only used here by
# generate_fun_fact() (the backfill path), never by the main Sonnet pass.
_HAIKU_INPUT_PER_MTOK = 1.00
_HAIKU_OUTPUT_PER_MTOK = 5.00

# ── Affiliate link config (Phase 2) ──────────────────────────────────────────
# CJ Affiliate account — Gerry's Publisher ID + Booking.com's Advertiser/Link
# ID. Booking.com North America approval covers the deep-link format below.
_CJ_PUBLISHER_ID = "101819605"
_BOOKING_LINK_ID = "17293132"
_CJ_BASE_URL = f"https://www.anrdoezrs.net/click-{_CJ_PUBLISHER_ID}-{_BOOKING_LINK_ID}"


def _booking_affiliate_url(search_query: str) -> str:
    """
    Booking.com search-results URL, wrapped in the CJ affiliate tracking
    link so hotel clicks are attributed to Gerry's account.
    NOTE: the real search path is /searchresults.html, not /search.html.
    """
    target = f"https://www.booking.com/searchresults.html?ss={urllib.parse.quote_plus(search_query)}"
    return f"{_CJ_BASE_URL}?url={urllib.parse.quote(target, safe='')}"


# Expedia Group Travel Creator Program — approved July 2026. camref/
# creativeref/adref are the fixed identifiers tied to Gerry's account and
# this generated link (confirmed via their Link Builder tool, not guessed);
# only `landingPage` changes per hotel/destination search.
_EXPEDIA_CAMREF = "1110lK3nQ"
_EXPEDIA_CREATIVEREF = "1100l68075"
_EXPEDIA_ADREF = "PZTaSwOiKr"


def _expedia_affiliate_url(search_query: str) -> str:
    """
    Expedia Hotel-Search URL, wrapped in Gerry's Expedia Group affiliate
    link (via their Travel Creator Program) so hotel clicks are attributed
    to his account.
    """
    target = f"https://www.expedia.com/Hotel-Search?destination={urllib.parse.quote_plus(search_query)}"
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
    city = destination.split(",")[0].strip() if destination else ""

    for day in data.get("days", []):
        for stop in day.get("stops", []):
            category = stop.get("category")
            name = (stop.get("name") or "").strip()
            is_specific = stop.get("is_specific_name", True)

            if category == "hotel":
                query = f"{name} {city}".strip() if is_specific else city
                stop["booking_url"] = _booking_affiliate_url(query)
                stop["expedia_url"] = _expedia_affiliate_url(query)
            elif category == "food":
                query = f"{name} {city}".strip()
                stop["booking_url"] = _google_maps_search_url(query)
            elif category in ("activity", "sight"):
                query = f"{name} {city}".strip() if is_specific else f"{city} tours activities"
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

Schema:
{
  "destination": "City, Country",
  "duration": "X days",
  "summary": "2-3 sentence intro: what makes this destination special, and what this specific route covers (e.g. its vibe, standout stops, or theme)",
  "fun_fact": "One genuinely true fact about the destination itself that's surprising, quirky, or slightly odd — the kind that makes someone think 'huh, no way' or 'okay, now I want to go there.' NOT about any specific stop in this route, and NOT a flat postcard statement. One sentence, max ~20 words.",
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
          "description": "One sentence — what makes it special",
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

"fun_fact" — one REAL, verifiable fact about the destination itself (the
city/region/country as a whole), never about a specific stop in this
itinerary. Aim for something that actually surprises people — a fact that
makes them go "wait, really?" and gives them a genuine reason to want to
visit, not a flat, generic statement. Good angles to look for: an unusual
law or local custom, a record the place quietly holds, a strange bit of
history, an oddly specific number, something counter-intuitive about its
size/age/location, or a "most people don't know this" detail. Bad
examples to avoid: "has beautiful beaches", "is known for its rich
culture", "is a popular tourist destination" — these say nothing. Never
invent or guess a fact — if you aren't confident an intriguing fact is
actually true, fall back to a more general but definitely-true fact
instead (e.g. a well-established geographic or historical fact) rather
than risk a wrong specific statistic. One sentence. No source citation
needed, just the fact itself.

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
- Always write "summary": 2-3 sentences, no more — first sentence about
  what makes the destination itself appealing, second (and optionally
  third) about what this specific route covers or its vibe. Plain,
  engaging prose — not a bullet list, not marketing hype.
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


def analyse_frames(frame_paths: list[str], comments: list[dict] | None = None) -> tuple[Itinerary, str, list[str], float]:
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

    data["fun_fact"] = str(data.get("fun_fact") or "").strip()
    data["travel_tips"] = [str(t).strip() for t in (data.get("travel_tips") or []) if str(t).strip()]

    return Itinerary(**data), price_category, tags, cost_usd


_FUN_FACT_SYSTEM = """You write one short, genuinely true, surprising fun fact about a travel destination for a travel app's homepage — the kind that makes someone think "huh, no way" or "okay, now I actually want to go there."

Rules:
- Must be a REAL, verifiable fact — never invent or guess. If you are not confident a surprising fact is true, write a more general but still definitely-true fact instead.
- About the destination itself (the city/region/country) — not about any specific hotel, restaurant, or attraction.
- Favor quirky, little-known, or counter-intuitive angles: an unusual law or custom, a record the place quietly holds, a strange bit of history, an oddly specific number, a "most people don't know this" detail.
- Avoid flat, generic filler — sentences like "has beautiful beaches", "is known for its rich culture", or "is a popular tourist destination" say nothing and should never be the output.
- One sentence, max ~20 words.
- No markdown, no surrounding quotes — just the sentence itself.

Reply with ONLY the fact sentence, nothing else."""


def generate_fun_fact(destination: str) -> tuple[str, float]:
    """
    Cheap, standalone Haiku call that returns one real, general-knowledge
    fun fact about `destination` (e.g. "Rome, Italy") — used to backfill
    `fun_fact` on routes that were generated before that field existed,
    without re-running the full (much more expensive) Sonnet video
    analysis. Brand-new routes get their fun_fact for free as part of the
    same analyse_frames() call above — this function exists only for that
    backfill path (see main.py's startup backfill + /admin/backfill-fun-facts).

    Returns (fact, cost_usd). Never raises — returns ("", 0.0) on any
    failure (rate limit, bad response, network error) so a backfill run
    over many routes can skip this one and keep going rather than aborting
    the whole batch.
    """
    try:
        response = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            system=_FUN_FACT_SYSTEM,
            messages=[{"role": "user", "content": f"Destination: {destination}"}],
        )
        usage = getattr(response, "usage", None)
        cost_usd = 0.0
        if usage:
            cost_usd = (
                usage.input_tokens * _HAIKU_INPUT_PER_MTOK
                + usage.output_tokens * _HAIKU_OUTPUT_PER_MTOK
            ) / 1_000_000
        fact = response.content[0].text.strip().strip('"').strip()
        return fact, cost_usd
    except Exception as e:
        print(f"[FunFact] generate_fun_fact failed for '{destination}': {e}")
        return "", 0.0


_TRIP_SUMMARY_SYSTEM = """You write a short intro for a travel app's route page — the same "summary" a video-extracted route gets, just for a traveler who built their own trip instead of pasting a video link.

You'll receive a destination and the list of specific attractions/activities the traveler picked for their trip.

Write 2-3 sentences, no more: the first about what makes the destination itself appealing, the second (and optionally third) about what THIS specific selection of stops covers or its vibe — reference the actual picks where it reads naturally, not just the destination in the abstract. Plain, engaging prose — not a bullet list, not marketing hype, not generic filler like "has beautiful beaches" or "is a popular tourist destination".

Reply with ONLY the summary text, nothing else — no markdown, no surrounding quotes."""


def generate_trip_summary(destination: str, stop_names: list[str]) -> tuple[str, float]:
    """
    Cheap, standalone Haiku call that writes the same kind of 2-3 sentence
    destination + route summary a video-extracted route gets for free from
    the main Sonnet call (see SYSTEM_PROMPT's "summary" field) — Build Your
    Own Trip has no video to extract that from, so this fills the same gap
    generate_fun_fact/generate_travel_tips fill for their own fields.

    Returns (summary, cost_usd). Never raises — returns ("", 0.0) on any
    failure, same non-fatal contract as generate_fun_fact.
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
    same role generate_fun_fact plays for fun_fact. New video-extracted
    routes get travel_tips for free as part of the same analyse_frames()
    call instead (see SYSTEM_PROMPT above).

    Returns (tips, cost_usd). Never raises — returns ([], 0.0) on any
    failure, same non-fatal contract as generate_fun_fact.
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
    generate_fun_fact, so the frontend can fall back to its own static-list
    matching rather than breaking the quiz.
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
