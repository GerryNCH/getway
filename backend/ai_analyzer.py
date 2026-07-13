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

import anthropic
from models import Itinerary

_client = anthropic.Anthropic()

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
          "is_specific_name": true
        }
      ]
    }
  ]
}

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
- Extract EVERY named location — do not summarise or skip stops
- If a place has no visible name after checking the clues above, describe it
  precisely and set "is_specific_name": false — e.g. "Cliffside restaurant
  with blue umbrellas, Positano"
- Group logically by day; if days unclear, distribute across Day 1
- If you see fewer than 3 identifiable locations, mention it in Day 1 label
- NEVER invent places you cannot see or read in the frames
- Return only the JSON object"""


def analyse_frames(frame_paths: list[str]) -> Itinerary:
    """
    Sends all frames to Claude Sonnet and parses the JSON itinerary response.
    Raises ValueError if the response cannot be parsed.
    """
    # Build multimodal content: intro text + all JPEG frames
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"Here are {len(frame_paths)} evenly-spaced frames from a travel video. "
                "Please identify every travel location and return the structured JSON itinerary."
            ),
        }
    ]

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

    return Itinerary(**data)
