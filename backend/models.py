"""
models.py — Shared Pydantic schemas used across all modules.
"""
from pydantic import BaseModel
from typing import Optional


class Stop(BaseModel):
    name: str
    category: str        # hotel | sight | food | activity | beach | village
    description: str
    tip: str = ""
    photo_url: str = "" # Google Places photo or empty string
    is_specific_name: bool = True  # False = AI couldn't confirm a real named
                                    # property/place — `name` is a generic
                                    # description, not a specific match.
                                    # Defaults True so cached itineraries
                                    # generated before this field existed
                                    # keep their old (assumed-specific) behavior.


class DayPlan(BaseModel):
    day: int
    label: str
    stops: list[Stop]


class Itinerary(BaseModel):
    destination: str
    duration: str
    days: list[DayPlan]
    hero_photo_url: str = ""  # Best single photo for the destination
    gallery_photo_urls: list[str] = []  # 4-5 photos for the hero gallery


class ExtractRequest(BaseModel):
    url: str
    max_frames: int = 8


class ExtractResponse(BaseModel):
    itinerary: Itinerary
    source: str          # "cache" | "ai_generated"
    video_id: str = ""
    cached: bool = False
