from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_
from typing import List, Optional
import time
from database import models, schemas
from database.database import get_db

router = APIRouter(prefix="/api/recommendations", tags=["Hotspot Recommendations"])

# ── Global Cache for Attractions (TTL 5 minutes) ──
_attractions_cache = {}
_ATTRACTIONS_TTL = 300

def _get_cached_attractions(cache_key: str):
    entry = _attractions_cache.get(cache_key)
    if entry and time.time() - entry["ts"] < _ATTRACTIONS_TTL:
        return entry["data"]
    return None

def _set_cached_attractions(cache_key: str, data):
    _attractions_cache[cache_key] = {"data": data, "ts": time.time()}



@router.get("", response_model=schemas.PaginatedAttractionResponse)
def list_attractions(
    city: Optional[str] = Query(None, description="Filter by city name"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """
    List all attractions with optional city/category filtering.
    Returns attractions with their parent city name for display.
    Ordered by rating DESC; duplicates (same name + category) are removed,
    keeping the entry from the best-matched city.
    """
    cache_key = f"{city}_{category}_{limit}_{offset}"
    cached = _get_cached_attractions(cache_key)
    if cached is not None:
        return cached

    query = (
        db.query(models.Attraction)
        .join(models.City)
        .order_by(models.Attraction.rating.desc())
    )

    if city:
        query = query.filter(models.City.name.ilike(f"%{city}%"))

    if category:
        query = query.filter(models.Attraction.category.ilike(f"%{category}%"))

    # Get count before limit/offset for pagination UI
    total_raw = query.count()

    attractions = query.offset(offset).limit(limit).all()

    # Build response with city_name and fallback lat/lng from parent city.
    seen = set()
    result = []
    for attr in attractions:
        dedup_key = (attr.name.strip().lower(), (attr.category or "").strip().lower())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        data = {
            "id": attr.id,
            "city_id": attr.city_id,
            "name": attr.name,
            "category": attr.category,
            "rating": attr.rating,
            "latitude": attr.latitude if attr.latitude else (attr.city.latitude if attr.city else None),
            "longitude": attr.longitude if attr.longitude else (attr.city.longitude if attr.city else None),
            "description": attr.description,
            "city_name": attr.city.name if attr.city else None,
            "image_url": attr.image_url,
        }
        result.append(data)

    # If the DB returned exactly 'limit' rows, there's likely more data to fetch, 
    # even if deduplication made the current batch smaller than 'limit'.
    has_more = len(attractions) == limit

    response_data = {
        "items": result,
        "total": total_raw,
        "has_more": has_more,
        "offset": offset,
        "limit": limit
    }
    _set_cached_attractions(cache_key, response_data)
    return response_data


@router.get("/search", response_model=List[schemas.CityWithDetails])
def search_hotspots(
    query: str = Query(..., description="Search by city or attraction name"),
    db: Session = Depends(get_db),
):
    """
    Search for cities or specific attractions.
    Returns the city details along with its attractions and safety info.
    """
    results = (
        db.query(models.City)
        .join(models.City.attractions, isouter=True)
        .filter(
            or_(
                models.City.name.ilike(f"%{query}%"),
                models.Attraction.name.ilike(f"%{query}%"),
            )
        )
        .distinct()
        .all()
    )

    if not results:
        raise HTTPException(status_code=404, detail="No matching hotspots found.")

    return results