"""
models.py — Shared Pydantic schemas used across all modules.
"""
from pydantic import BaseModel
from typing import Optional


class UnsplashAttribution(BaseModel):
    """
    Required by Unsplash's API guidelines whenever a photo they returned is
    displayed: the photographer's name + profile link, and a link to the
    photo's own Unsplash page. Only set when a photo actually came from
    Unsplash — Google Places photos use a different license and don't need
    this.
    """
    photographer_name: str = ""
    photographer_url: str = ""
    unsplash_url: str = ""


class Stop(BaseModel):
    name: str
    category: str        # hotel | sight | food | activity | beach | village
    description: str
    tip: str = ""
    photo_url: str = "" # Google Places photo or empty string
    photo_attribution: Optional[UnsplashAttribution] = None  # set only when
                                                               # photo_url came from Unsplash
    is_specific_name: bool = True  # False = AI couldn't confirm a real named
                                    # property/place — `name` is a generic
                                    # description, not a specific match.
                                    # Defaults True so cached itineraries
                                    # generated before this field existed
                                    # keep their old (assumed-specific) behavior.
    booking_url: str = ""  # Affiliate/booking link generated server-side in
                            # ai_analyzer.py: Booking.com (via CJ affiliate)
                            # for hotel stops, Google Maps search for food
                            # stops. Empty string for itineraries cached
                            # before this field existed — frontend falls
                            # back to building its own (non-affiliate) link.
    expedia_url: str = ""  # Expedia affiliate link (Travel Creator Program),
                            # hotel stops only. Empty for non-hotel stops and
                            # for itineraries cached before this field existed.


class DayPlan(BaseModel):
    day: int
    label: str
    stops: list[Stop]


class Comment(BaseModel):
    text: str
    username: str = ""
    likes: int = 0
    reply_count: int = 0
    avatar_url: str = ""
    created_at: str = ""


class Itinerary(BaseModel):
    destination: str
    duration: str
    days: list[DayPlan]
    summary: str = ""  # 2-3 sentence intro: what makes the destination
                        # special + what this particular route covers.
                        # Empty for itineraries cached before this field
                        # existed.
    hero_photo_url: str = ""  # Best single photo for the destination
    hero_attribution: Optional[UnsplashAttribution] = None
    gallery_photo_urls: list[str] = []  # 4-5 photos for the hero gallery
    gallery_attributions: list[UnsplashAttribution] = []  # parallel to gallery_photo_urls
    comments: list[Comment] = []  # Real TikTok comments, fetched via Apify
                                   # (clockworks/tiktok-comments-scraper).
                                   # Empty for itineraries cached before this
                                   # field existed, or if the fetch failed
                                   # (non-fatal — see extractor.py).


class RouteMeta(BaseModel):
    price_category: str = "€€"   # "€" | "€€" | "€€€"
    tags: list[str] = []          # most_popular | luxury | budget_friendly |
                                   # exotic | mountain | city | beach
    creator_handle: str = ""      # e.g. "@username"


class ExtractRequest(BaseModel):
    url: str
    max_frames: int = 8


class ExtractResponse(BaseModel):
    itinerary: Itinerary
    source: str          # "cache" | "ai_generated"
    video_id: str = ""
    cached: bool = False


class ReviewCreate(BaseModel):
    video_id: str
    name: str
    title: str
    rating: int
    text: str


class Review(BaseModel):
    id: int
    video_id: str
    name: str
    title: str
    rating: int
    text: str
    created_at: str


class ReviewsResponse(BaseModel):
    reviews: list[Review]
    average_rating: float = 0.0
    count: int = 0
