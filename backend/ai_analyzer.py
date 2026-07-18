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


def _attach_booking_urls(data: dict) -> dict:
    """
    Fills in `booking_url` (Booking.com or Maps) and `expedia_url` (hotels
    only) for every hotel/food stop, in place, before the JSON is turned
    into an Itinerary:
      - hotel → Booking.com search wrapped in the CJ affiliate link, PLUS
        an Expedia affiliate search as a second option. Uses the exact stop
        name when the AI confirmed it (`is_specific_name`), otherwise falls
        back to just the destination city — same rule the frontend already
        applies for its own "unconfirmed hotel" fallback.
      - food  → Google Maps search (name + city) as a stand-in until
        TheFork affiliate is live.
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
  "price_category": "€ | €€ | €€€",
  "tags": ["0-3 of: most_popular, luxury, budget_friendly, exotic, mountain, city, beach"],
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

    return Itinerary(**data), price_category, tags, cost_usd
