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
    similar_hotel_is_real: bool = False  # Hotel stops only, and only when
                                           # is_specific_name is False. True
                                           # when places.py upgraded an
                                           # AI-written description (e.g.
                                           # "Beachfront resort, Cala Sant
                                           # Vicenç area") to an actual real,
                                           # bookable hotel found via Google
                                           # Places — so stop.name is now a
                                           # real property name, not a
                                           # generic description, and a real
                                           # "Book this hotel" link is
                                           # possible even though it's still
                                           # not confirmed as the exact one
                                           # shown in the video.
    lat: Optional[float] = None  # Stop coordinates for the route Map view
    lng: Optional[float] = None  # (Leaflet, index.html). Populated from
                                   # Google Places' geometry.location when
                                   # available — see places.py. None for
                                   # itineraries cached before this field
                                   # existed, or when Places couldn't
                                   # resolve the stop; the frontend hides
                                   # the Map view entirely for a day with
                                   # no stop coordinates rather than
                                   # showing a broken/empty map.
    is_free: bool = False  # Build Your Own Trip stops only — carried
                             # through from TripCandidate.is_free. Video-
                             # extracted stops always default False (no
                             # equivalent classification runs on that path).
                             # The frontend uses this to suppress the
                             # GetYourGuide ticket button on free public
                             # places (parks, plazas, viewpoints).
    estimated_price: str = ""  # Build Your Own Trip stops only — carried
                                 # through from TripCandidate.estimated_price
                                 # (e.g. "€16"). Empty for video-extracted
                                 # stops and whenever unknown/free.
    is_full_day: bool = False  # Build Your Own Trip stops only — carried
                                 # through from TripCandidate.is_full_day.
                                 # Always False for video-extracted stops.


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
    source_url: str = ""  # original TikTok/Instagram clip URL — populated from
                            # the DB's `url` column so the frontend can always
                            # link back to the source video, regardless of
                            # which endpoint loaded the route.
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
    car_rental_recommended: bool = False  # AI-set (and admin-overridable):
                                            # true when the video itself
                                            # shows/mentions driving, or the
                                            # itinerary covers multiple towns
                                            # best reached by car. Shows a
                                            # "Rent a car" box under the
                                            # hotel box when true.
    car_rental_note: str = ""  # Short reason shown alongside the car
                                 # rental box, e.g. "This itinerary covers
                                 # several towns best explored by car."
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
    fun_fact: str = ""  # One short, real, verifiable fact about the
                          # destination itself (not the specific stops) —
                          # e.g. "Rome has more fountains than any other
                          # city in the world." Written by Claude: for new
                          # routes it comes free as part of the same
                          # multimodal analysis call (see ai_analyzer.py's
                          # SYSTEM_PROMPT); for routes generated before this
                          # field existed, it's backfilled with a separate,
                          # cheap Haiku call (ai_analyzer.generate_fun_fact),
                          # which main.py runs automatically once at startup
                          # and can also be re-triggered manually via
                          # POST /admin/backfill-fun-facts. Empty until
                          # filled — the homepage fact chip (index.html)
                          # just doesn't render for that route until it is.


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


# ── Build Your Own Trip (Phase A: candidate search) ─────────────────────────

class TripCandidate(BaseModel):
    name: str
    description: str
    category: str  # sight | food | activity | beach | village — same values
                    # as Stop.category; never "hotel" (hotels are handled
                    # separately, by locked logic — see trip_builder.py).
    photo_url: str = ""
    rating: float = 0.0
    user_rating_count: int = 0
    price_level: str = ""  # Google's raw Places priceLevel enum, e.g.
                             # "PRICE_LEVEL_MODERATE" — empty if Google has
                             # no price signal for this place. NOTE: a
                             # free-by-type place (see is_free below) also
                             # has price_level="" — Google doesn't set
                             # PRICE_LEVEL_FREE for parks/plazas/monuments,
                             # so price_level alone can't distinguish "free"
                             # from "no price data available". Use is_free
                             # for that.
    is_free: bool = False  # True = classified free by place type (park,
                             # plaza, outdoor monument, etc. — see
                             # trip_builder._is_free_by_type), regardless of
                             # what price_level says.
    lat: Optional[float] = None
    lng: Optional[float] = None
    is_famous: bool = False  # True = included regardless of budget tier
                               # (see trip_builder.fits_budget) because it's
                               # a must-see by rating x review count.
    section: str = "attraction"  # "attraction" | "activity" — which search
                                   # this candidate came from: the plain
                                   # destination-wide attraction search, or
                                   # one of the traveler's requested activity
                                   # types (places._ACTIVITY_TYPE_QUERY_TERMS).
                                   # Lets the checklist UI group results
                                   # instead of showing one flat list.
    estimated_price: str = ""  # Short human string, e.g. "€16" or "€10-15" —
                                 # the AI curation pass's own ballpark price
                                 # estimate for paid attractions/activities/
                                 # tours it has reasonably confident general
                                 # knowledge of (see trip_builder's
                                 # _CURATION_SYSTEM). Empty when free or when
                                 # the AI isn't confident enough to avoid a
                                 # misleading guess — NOT the same as
                                 # price_level (Google's own signal).
    is_full_day: bool = False  # True = the AI curation pass judged this a
                                 # major attraction that typically takes most
                                 # or all of a day to properly visit (a theme
                                 # park, a day-trip island, a big safari/
                                 # wildlife park, a major hike, a full-day
                                 # fjord/boat cruise). trip_builder.
                                 # group_into_days gives each of these its
                                 # OWN day rather than packing other stops
                                 # alongside it — same rule
                                 # ai_analyzer.py's video-extraction prompt
                                 # already applies to AI-generated routes.


class TripCandidatesRequest(BaseModel):
    destination: str
    budget: str  # cheap | mid | luxury
    activity_types: list[str] = []  # optional slugs from
                                      # places._ACTIVITY_TYPE_QUERY_TERMS
                                      # (e.g. "nightlife", "history") — empty
                                      # (the default) means the old
                                      # attractions-only behavior, unchanged.


class TripCandidatesResponse(BaseModel):
    destination: str
    budget: str
    candidates: list[TripCandidate]
    cached: bool = False


class TripFunFactRequest(BaseModel):
    destination: str


class TripFunFactResponse(BaseModel):
    fun_fact: str = ""  # One real fact about the destination itself (see
                          # ai_analyzer.generate_fun_fact). Deliberately its
                          # own endpoint (POST /trip/fun-fact), not part of
                          # /trip/candidates: it's fast (one cheap Haiku
                          # call, no Places search), so the frontend fires
                          # it in parallel with the slower candidates call
                          # and can show it during the "finding places"
                          # spinner instead of only after candidates finish
                          # loading. Empty on failure (non-fatal).


# ── Build Your Own Trip (Phase B: hotel recommendation) ─────────────────────

class SelectedAttraction(BaseModel):
    """
    One attraction the traveler checked off from their TripCandidate list —
    just enough (name + coordinates) to anchor the hotel search
    geographically. See trip_builder.cluster_center().
    """
    name: str
    lat: float
    lng: float


class TripHotelRequest(BaseModel):
    destination: str
    budget: str  # cheap | mid | luxury
    selected_attractions: list[SelectedAttraction]  # anchors the hotel search — see trip_builder.cluster_center()


class TripHotelRecommendation(BaseModel):
    name: str
    description: str = ""
    photo_url: str = ""
    rating: float = 0.0
    user_rating_count: int = 0
    price_level: str = ""  # Google's raw Places priceLevel enum — empty if unset
    property_type: str = ""  # Not populated by Places directly in Phase B —
                               # left empty for now, same convention as
                               # Stop.property_type when not confidently known.
    area_label: str = ""  # Same — left empty in Phase B.
    lat: Optional[float] = None
    lng: Optional[float] = None
    booking_url: str = ""  # Built via ai_analyzer._booking_affiliate_url — same
                             # affiliate link logic as video-extracted hotel stops.
    expedia_url: str = ""  # Built via ai_analyzer._expedia_affiliate_url.


class TripHotelResponse(BaseModel):
    destination: str
    budget: str
    hotel: Optional[TripHotelRecommendation] = None  # None = no open, 4.0+-rated
                                                        # hotel found near the
                                                        # selected attractions.
    cached: bool = False


# ── Build Your Own Trip (Phase C: itinerary assembly) ───────────────────────

class TripBuildRequest(BaseModel):
    destination: str
    days: int
    people: int = 1
    budget: str  # cheap | mid | luxury
    selected_attractions: list[TripCandidate]  # the candidates the traveler
                                                 # checked off from
                                                 # /trip/candidates — reused
                                                 # as-is (name/description/
                                                 # category/photo_url/lat/lng)
                                                 # so Phase C needs no extra
                                                 # Places lookups.


# ── Build Your Own Trip (Phase D: save + share) ──────────────────────────────

class TripSaveRequest(BaseModel):
    destination: str
    days: int
    people: int = 1
    budget: str  # cheap | mid | luxury
    selected_attractions: list[TripCandidate]  # kept alongside the built
                                                 # itinerary so the trip can
                                                 # later be reopened in the
                                                 # builder — see edit-state.
    itinerary: Itinerary  # the built snapshot, as returned by /trip/build
    slug: Optional[str] = None  # pass an existing slug to update that saved
                                  # trip in place (the edit-and-resave flow);
                                  # omit to save a brand-new trip.


class TripSaveResponse(BaseModel):
    slug: str


class TripEditStateResponse(BaseModel):
    slug: str
    destination: str
    days: int
    people: int
    budget: str
    selected_attractions: list[TripCandidate]


# ── Build Your Own Trip (manual search-and-add) ──────────────────────────────

class TripSearchRequest(BaseModel):
    destination: str
    query: str  # free-text, traveler-typed (e.g. "diving") — for something
                 # the curated candidate lists didn't surface. Not budget-
                 # filtered: the traveler explicitly asked for this, so
                 # trip_builder.fits_budget is deliberately not applied here.


class TripSearchResponse(BaseModel):
    destination: str
    query: str
    candidates: list[TripCandidate]
