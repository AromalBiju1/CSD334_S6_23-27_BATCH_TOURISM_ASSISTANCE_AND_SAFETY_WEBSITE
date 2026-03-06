"""
Personalized Recommendations Router
ML-based content filtering: recommends attractions based on user preferences,
visited places, and category similarity scoring.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func
from typing import List
from database import models, schemas
from database.database import get_db
from routers.auth.deps import get_current_user
from collections import Counter
import math

router = APIRouter(prefix="/api/user-recommendations", tags=["Personalized Recommendations"])


# ═══════════════════════════════════════════════
#  PREFERENCES CRUD
# ═══════════════════════════════════════════════

@router.get("/preferences", response_model=schemas.UserPreferenceResponse)
def get_preferences(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get current user's travel preferences."""
    pref = db.query(models.UserPreference).filter(
        models.UserPreference.user_id == current_user.id
    ).first()
    
    if not pref:
        # Create default preferences
        pref = models.UserPreference(user_id=current_user.id)
        db.add(pref)
        db.commit()
        db.refresh(pref)
    
    return pref


@router.put("/preferences", response_model=schemas.UserPreferenceResponse)
def update_preferences(
    data: schemas.UserPreferenceCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Update user's travel preferences."""
    pref = db.query(models.UserPreference).filter(
        models.UserPreference.user_id == current_user.id
    ).first()
    
    if not pref:
        pref = models.UserPreference(user_id=current_user.id)
        db.add(pref)
    
    pref.preferred_categories = data.preferred_categories
    pref.budget_level = data.budget_level
    pref.travel_style = data.travel_style
    pref.preferred_safety = data.preferred_safety
    
    db.commit()
    db.refresh(pref)
    return pref


# ═══════════════════════════════════════════════
#  VISITED PLACES CRUD
# ═══════════════════════════════════════════════

@router.get("/visited", response_model=List[schemas.VisitedPlaceResponse])
def get_visited_places(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get all places the user has visited."""
    visited = (
        db.query(models.VisitedPlace)
        .filter(models.VisitedPlace.user_id == current_user.id)
        .all()
    )
    
    result = []
    for v in visited:
        attr = db.query(models.Attraction).filter(models.Attraction.id == v.attraction_id).first()
        city = db.query(models.City).filter(models.City.id == attr.city_id).first() if attr else None
        result.append({
            "id": v.id,
            "attraction_id": v.attraction_id,
            "rating": v.rating,
            "visited_at": v.visited_at,
            "attraction_name": attr.name if attr else None,
            "attraction_category": attr.category if attr else None,
            "city_name": city.name if city else None,
        })
    
    return result


@router.post("/visited", response_model=schemas.VisitedPlaceResponse)
def add_visited_place(
    data: schemas.VisitedPlaceCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Mark a place as visited."""
    # Check attraction exists
    attr = db.query(models.Attraction).filter(models.Attraction.id == data.attraction_id).first()
    if not attr:
        raise HTTPException(status_code=404, detail="Attraction not found")
    
    # Check not already visited
    existing = db.query(models.VisitedPlace).filter(
        models.VisitedPlace.user_id == current_user.id,
        models.VisitedPlace.attraction_id == data.attraction_id,
    ).first()
    
    if existing:
        existing.rating = data.rating
        db.commit()
        db.refresh(existing)
        vp = existing
    else:
        vp = models.VisitedPlace(
            user_id=current_user.id,
            attraction_id=data.attraction_id,
            rating=data.rating,
        )
        db.add(vp)
        db.commit()
        db.refresh(vp)
    
    city = db.query(models.City).filter(models.City.id == attr.city_id).first()
    return {
        "id": vp.id,
        "attraction_id": vp.attraction_id,
        "rating": vp.rating,
        "visited_at": vp.visited_at,
        "attraction_name": attr.name,
        "attraction_category": attr.category,
        "city_name": city.name if city else None,
    }


@router.delete("/visited/{place_id}")
def remove_visited_place(
    place_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Remove a visited place."""
    vp = db.query(models.VisitedPlace).filter(
        models.VisitedPlace.id == place_id,
        models.VisitedPlace.user_id == current_user.id,
    ).first()
    
    if not vp:
        raise HTTPException(status_code=404, detail="Visited place not found")
    
    db.delete(vp)
    db.commit()
    return {"message": "Removed from visited places"}


# ═══════════════════════════════════════════════
#  ML RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════

def _compute_category_similarity(user_categories: list, attraction_category: str) -> float:
    """Content-based category matching using TF-like scoring."""
    if not attraction_category or not user_categories:
        return 0.0
    
    attr_cat = attraction_category.lower().strip()
    for uc in user_categories:
        if uc.lower().strip() == attr_cat:
            return 1.0
        # Partial match (e.g., "Historical" partially matches "Historical Monument")
        if uc.lower().strip() in attr_cat or attr_cat in uc.lower().strip():
            return 0.7
    
    return 0.0


def _compute_safety_score(safety_zone: str, preferred_safety: str) -> float:
    """Score based on user's safety preference."""
    if preferred_safety == "all":
        return 0.5  # Neutral 
    
    zone_scores = {"green": 1.0, "orange": 0.5, "red": 0.1}
    if preferred_safety == "green" and safety_zone == "green":
        return 1.0
    elif preferred_safety == "green" and safety_zone == "orange":
        return 0.3
    elif preferred_safety == "green" and safety_zone == "red":
        return 0.0
    
    return zone_scores.get(safety_zone, 0.3)


def _compute_collaborative_boost(attraction_id: int, visited_ids: set, 
                                  all_visited: list, db: Session) -> float:
    """
    Simple collaborative filtering: if other users who visited similar
    attractions also visited this one, boost its score.
    """
    if not visited_ids:
        return 0.0
    
    # Find users who visited the same attractions as current user
    similar_users = (
        db.query(models.VisitedPlace.user_id)
        .filter(
            models.VisitedPlace.attraction_id.in_(visited_ids),
            models.VisitedPlace.user_id != all_visited[0].user_id if all_visited else True
        )
        .distinct()
        .limit(50)
        .all()
    )
    
    if not similar_users:
        return 0.0
    
    similar_user_ids = [u[0] for u in similar_users]
    
    # Count how many similar users visited this attraction
    count = (
        db.query(func.count(models.VisitedPlace.id))
        .filter(
            models.VisitedPlace.attraction_id == attraction_id,
            models.VisitedPlace.user_id.in_(similar_user_ids)
        )
        .scalar()
    )
    
    # Normalize: more similar users who visited → higher boost
    return min(count / max(len(similar_user_ids), 1), 1.0) * 0.3


@router.get("/for-you", response_model=List[schemas.RecommendationResponse])
def get_recommendations(
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    ML-based personalized recommendations.
    
    Algorithm:
    1. Get user preferences and visited places
    2. For each unvisited attraction, compute a composite score:
       - Category similarity (content-based filtering): 40% weight
       - Rating quality: 20% weight
       - Safety preference match: 20% weight
       - Collaborative filtering boost: 20% weight
    3. Sort by composite score descending
    4. Return top N recommendations
    """
    # 1. Get user preferences
    pref = db.query(models.UserPreference).filter(
        models.UserPreference.user_id == current_user.id
    ).first()
    
    user_categories = []
    preferred_safety = "all"
    if pref:
        user_categories = [c.strip() for c in pref.preferred_categories.split(",") if c.strip()]
        preferred_safety = pref.preferred_safety
    
    # 2. Get visited attraction IDs
    visited = db.query(models.VisitedPlace).filter(
        models.VisitedPlace.user_id == current_user.id
    ).all()
    visited_ids = {v.attraction_id for v in visited}
    
    # If user has visited places but no explicit preferences,
    # infer categories from visited places
    if not user_categories and visited_ids:
        visited_attrs = db.query(models.Attraction.category).filter(
            models.Attraction.id.in_(visited_ids)
        ).all()
        cat_counts = Counter(a.category for a in visited_attrs if a.category)
        user_categories = [cat for cat, _ in cat_counts.most_common(5)]
    
    # 3. Get all candidate attractions (exclude already visited)
    candidates = (
        db.query(models.Attraction)
        .join(models.City)
        .filter(~models.Attraction.id.in_(visited_ids) if visited_ids else True)
        .all()
    )
    
    # 4. Score each candidate
    scored = []
    for attr in candidates:
        city = attr.city
        
        # Content-based: category similarity
        cat_score = _compute_category_similarity(user_categories, attr.category)
        
        # Rating quality (normalized 0-1)
        rating_score = (attr.rating or 0) / 5.0
        
        # Safety preference
        safety_score = _compute_safety_score(
            city.safety_zone if city else "orange",
            preferred_safety
        )
        
        # Collaborative filtering
        collab_score = _compute_collaborative_boost(attr.id, visited_ids, visited, db)
        
        # Weighted composite score
        composite = (
            cat_score * 0.40 +
            rating_score * 0.20 +
            safety_score * 0.20 +
            collab_score * 0.20
        )
        
        scored.append({
            "id": attr.id,
            "name": attr.name,
            "category": attr.category,
            "rating": attr.rating,
            "city_name": city.name if city else None,
            "city_id": attr.city_id,
            "safety_zone": city.safety_zone if city else None,
            "match_score": round(composite, 3),
            "latitude": attr.latitude or (city.latitude if city else None),
            "longitude": attr.longitude or (city.longitude if city else None),
            "description": attr.description,
        })
    
    # 5. Sort by composite score (descending), then rating
    scored.sort(key=lambda x: (x["match_score"], x["rating"] or 0), reverse=True)
    
    return scored[:limit]


@router.get("/categories")
def get_available_categories(db: Session = Depends(get_db)):
    """Get all unique attraction categories for the preference form."""
    categories = (
        db.query(models.Attraction.category)
        .filter(models.Attraction.category.isnot(None))
        .distinct()
        .all()
    )
    return [c[0] for c in categories if c[0]]


@router.get("/all-attractions")
def get_all_attractions_for_marking(
    search: str = "",
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get attractions for the 'Mark as Visited' search."""
    query = db.query(models.Attraction).join(models.City)
    
    if search:
        query = query.filter(
            models.Attraction.name.ilike(f"%{search}%") |
            models.City.name.ilike(f"%{search}%")
        )
    
    attractions = query.limit(limit).all()
    
    return [
        {
            "id": a.id,
            "name": a.name,
            "category": a.category,
            "rating": a.rating,
            "city_name": a.city.name if a.city else None,
        }
        for a in attractions
    ]
