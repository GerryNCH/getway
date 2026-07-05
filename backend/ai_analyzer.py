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
          "tip": "Practical tip from the creator, max 12 words (empty string if none)"
        }
      ]
    }
  ]
}

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

Rules:
- Extract EVERY named location — do not summarise or skip stops
- If a place has no visible name, describe it precisely: "Cliffside restaurant with blue umbrellas, Positano"
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
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    raw = response.content[0].text.strip()

    # Strip accidental markdown code fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    data = json.loads(raw)
    return Itinerary(**data)
