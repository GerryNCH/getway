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
    photo_is_fallback: bool = False  # True when photo_url came from the
                                       # generic category search (e.g. "city
                                       # adventure activity") rather than a
                                       # confident name-based match — flagged
                                       # in the admin panel so it's clear
                                       # which photos most need a manual look.
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
    property_type: str = ""  # Hotel stops only, e.g. "Boutique Hotel",
                               # "Beach Resort", "Guesthouse" — the AI's own
                               # read of the property's style from what's
                               # visible. Deliberately NOT a numeric rating
                               # (e.g. "8.9 Excellent") — we have no access to
                               # real Booking.com review scores, and inventing
                               # one would be showing a fabricated number as
                               # if it were a genuine guest rating.
    area_label: str = ""  # Hotel stops only, e.g. "Old Town", "Beachfront" —
                            # the neighbourhood/area, when identifiable.
    transfer_note: str = ""  # How to get here relative to the hotel/previous
                               # stop, e.g. "Short walk from Old Town", "Boat
                               # transfer needed" — kept qualitative rather
                               # than inventing precise minutes the AI can't
                               # actually know. Empty when not confidently
                               # inferable.
    maps_url_override: str = ""  # Admin-set fallback for when Google's own
                                   # search can't resolve the auto-built
                                   # name+city query (e.g. small streets,
                                   # region-biased accounts). Accepts either
                                   # a full Google Maps URL (pasted directly
                                   # from the address bar after manually
                                   # finding the place) or a plain address —
                                   # the frontend uses it as-is if it starts
                                   # with "http", otherwise wraps it in a
                                   # Maps search URL. Empty = auto-built as
                                   # before.


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
    creator_handle: str = ""  # TikTok @handle of the video's creator, e.g.
                               # "@username" — populated from yt-dlp's
                               # "uploader" field right after generation
                               # (see main.py), or set manually in the
                               # admin panel. Shown as a profile link on
                               # the route page so creators get credited.
    price_category: str = ""  # "€" | "€€" | "€€€" — the AI's own estimate
                                # (see ai_analyzer.py), shown as a pill next
                                # to the destination title, same as the
                                # static Mallorca demo route.
    generation_cost_usd: float = 0.0  # Real $ cost of generating this route
                                        # (troll filter + Sonnet analysis),
                                        # computed from Anthropic's actual
                                        # token usage — not an estimate.
    hotel_banner_photo_url: str = ""  # Admin-editable photo for the generic
                                        # "Hotels in [city]" fallback banner
                                        # shown when no real hotel stop was
                                        # found in the video. Falls back to
                                        # the last gallery photo if empty.
    view_count: int = 0            # Times the route page has been opened
    affiliate_click_count: int = 0  # Times a Booking/Expedia/Airbnb link
                                      # was clicked from this route
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


class SiteSettings(BaseModel):
    hero_slides: list[str] = []          # Homepage rotating background images
    featured_route_ids: list[str] = []    # Ordered video_ids to show on the
                                            # homepage grid; empty = show all
                                            # approved routes automatically
                                            # (original default behavior)


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
