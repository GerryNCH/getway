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


class DayPlan(BaseModel):
    day: int
    label: str
    stops: list[Stop]


class Itinerary(BaseModel):
    destination: str
    duration: str
    days: list[DayPlan]
    hero_photo_url: str = ""  # Google Places photo for the destination hero


class ExtractRequest(BaseModel):
    url: str
    max_frames: int = 8


class ExtractResponse(BaseModel):
    itinerary: Itinerary
    source: str          # "cache" | "ai_generated"
    video_id: str = ""
    cached: bool = False
