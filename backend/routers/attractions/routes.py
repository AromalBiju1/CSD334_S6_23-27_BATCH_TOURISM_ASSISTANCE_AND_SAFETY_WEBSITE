from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import List, Optional
from database import models, schemas
from database.database import get_db

router = APIRouter(prefix="/api/recommendations", tags=["Hotspot Recommendations"])

@router.get("/search", response_model=List[schemas.CityWithDetails])
def search_hotspots(
    query: str = Query(..., description="Search by city or attraction name"),
    db: Session = Depends(get_db)
):
    """
    Search for cities or specific attractions. 
    Returns the city details along with its attractions and safety info.
    """
    # Look for cities that match the query OR cities that have attractions matching the query
    results = db.query(models.City).join(models.City.attractions, isouter=True).filter(
        or_(
            models.City.name.ilike(f"%{query}%"),
            models.Attraction.name.ilike(f"%{query}%")
        )
    ).distinct().all()

    if not results:
        raise HTTPException(status_code=404, detail="No matching hotspots found.")

    return results