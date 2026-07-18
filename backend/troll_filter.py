"""
troll_filter.py — Cheap pre-screening before we spend money on multimodal AI.

Uses Claude Haiku (the fastest, cheapest model) with ONLY the video title +
description — no download needed. Cost: ~$0.0003 per check (300x cheaper
than a full multimodal analysis).

Rejects:
  - Cat/pet/funny videos
  - Music/cooking/fitness/gaming content
  - Spam or nonsensical input
  - Anything without a clear geographic destination
"""

import anthropic
from database import get_troll_decision, save_troll_decision

_client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env

# Claude Haiku 4.5 standard rate, $ per million tokens (verified July 2026 —
# update these two numbers if Anthropic changes pricing or this model is
# swapped for a different one).
_HAIKU_INPUT_PER_MTOK = 1.00
_HAIKU_OUTPUT_PER_MTOK = 5.00

TROLL_SYSTEM = """You are a travel content classifier.
Given a video title and description, decide if this is a TRAVEL video
that shows real geographic destinations a tourist could visit.

Reply with ONLY a JSON object:
{"is_travel": true/false, "reason": "one sentence"}

is_travel = true  → video clearly shows travel destinations (cities, beaches,
                     restaurants, hotels, landmarks, day trips, etc.)
is_travel = false → anything else (pets, cooking at home, gym, music,
                     gaming, random vlogs with no destinations, spam)

Be generous: if there's any reasonable doubt, set is_travel = true.
Only reject clearly non-travel content."""


def check_is_travel(video_id: str, title: str, description: str) -> tuple[bool, str, float]:
    """
    Returns (is_travel, reason, cost_usd).

    Checks the troll cache first — if we've seen this video before, returns
    the cached decision at zero cost (cost_usd = 0.0, no new API call made).
    """
    # Layer 1: check cache (free)
    cached = get_troll_decision(video_id)
    if cached is not None:
        return cached, "cached decision", 0.0

    # Layer 2: call Claude Haiku (cheap)
    text_to_check = f"Title: {title}\n\nDescription: {description[:600]}"

    cost_usd = 0.0
    try:
        response = _client.messages.create(
            model="claude-haiku-4-5-20251001",   # cheapest model
            max_tokens=100,
            system=TROLL_SYSTEM,
            messages=[{"role": "user", "content": text_to_check}],
        )
        usage = getattr(response, "usage", None)
        if usage:
            cost_usd = (
                usage.input_tokens * _HAIKU_INPUT_PER_MTOK
                + usage.output_tokens * _HAIKU_OUTPUT_PER_MTOK
            ) / 1_000_000
        import json
        raw = response.content[0].text.strip()
        result = json.loads(raw)
        is_travel = bool(result.get("is_travel", False))
        reason = result.get("reason", "")
    except Exception as e:
        # If the filter itself fails, allow through (don't block legitimate users)
        print(f"[TrollFilter] Error: {e} — allowing through")
        is_travel = True
        reason = f"filter error: {e}"

    # Save for next time (free for repeated requests)
    save_troll_decision(video_id, is_travel, reason)

    return is_travel, reason, cost_usd
