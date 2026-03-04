from fastapi import APIRouter, Depends, HTTPException, Query
<<<<<<< HEAD
from sqlalchemy.orm import Session
=======
from sqlalchemy.orm import Session, joinedload
>>>>>>> 7056f10a8e7bda46fe2dbc81fc1b2799fe0d581e
from sqlalchemy import or_
from typing import List, Optional
from database import models, schemas
from database.database import get_db

router = APIRouter(prefix="/api/recommendations", tags=["Hotspot Recommendations"])



@router.get("", response_model=List[schemas.AttractionWithCity])
def list_attractions(
    city: Optional[str] = Query(None, description="Filter by city name"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(500, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """
    List all attractions with optional city/category filtering.
    Returns attractions with their parent city name for display.
    Ordered by rating DESC; duplicates (same name + category) are removed,
    keeping the entry from the best-matched city.
    """
    query = (
        db.query(models.Attraction)
        .join(models.City)
        .order_by(models.Attraction.rating.desc())
    )

    if city:
        query = query.filter(models.City.name.ilike(f"%{city}%"))

    if category:
        query = query.filter(models.Attraction.category.ilike(f"%{category}%"))

    attractions = query.offset(offset).limit(limit).all()

    # Build response with city_name and fallback lat/lng from parent city.
    # Deduplicate by (name, category) — keep the first (highest-rated) occurrence
    # so that attractions seeded under multiple city rows appear only once.
    seen: set = set()
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
        }
        result.append(data)

    return result


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