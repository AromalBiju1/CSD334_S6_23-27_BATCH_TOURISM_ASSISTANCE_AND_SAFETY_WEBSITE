"""
Cities Router - CRUD operations for cities with safety data
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from database.database import get_db
from database import models, schemas

router = APIRouter(prefix="/api/cities", tags=["cities"])


@router.get("", response_model=List[schemas.CityResponse])
def get_cities(
    zone: Optional[str] = Query(None, description="Filter by safety zone (green, orange, red)"),
    state: Optional[str] = Query(None, description="Filter by state"),
    limit: int = Query(100, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """Get all cities with optional filtering."""
    query = db.query(models.City)
    
    if zone:
        query = query.filter(models.City.safety_zone == zone.lower())
    
    if state:
        query = query.filter(models.City.state.ilike(f"%{state}%"))
    
    cities = query.offset(offset).limit(limit).all()
    return cities


@router.get("/search", response_model=List[schemas.CityResponse])
def search_cities(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Search cities by name or state."""
    cities = db.query(models.City).filter(
        models.City.name.ilike(f"%{q}%") | 
        models.City.state.ilike(f"%{q}%")
    ).limit(limit).all()
    return cities


@router.get("/{city_id}", response_model=schemas.CityWithDetails)
def get_city(city_id: int, db: Session = Depends(get_db)):
    """Get city by ID with attractions and crime statistics."""
    city = db.query(models.City).filter(models.City.id == city_id).first()
    if not city:
        raise HTTPException(status_code=404, detail="City not found")
    return city


@router.get("/state/{state_name}", response_model=List[schemas.CityResponse])
def get_cities_by_state(state_name: str, db: Session = Depends(get_db)):
    """Get all cities in a specific state."""
    cities = db.query(models.City).filter(
        models.City.state.ilike(f"%{state_name}%")
    ).all()
    return cities


@router.get("/zone/{zone}", response_model=List[schemas.CityResponse])
def get_cities_by_zone(zone: str, db: Session = Depends(get_db)):
    """Get all cities with a specific safety zone."""
    if zone not in ["green", "orange", "red"]:
        raise HTTPException(status_code=400, detail="Invalid zone. Use green, orange, or red")
    
    cities = db.query(models.City).filter(models.City.safety_zone == zone).all()
    return cities


@router.post("", response_model=schemas.CityResponse)
def create_city(city: schemas.CityCreate, db: Session = Depends(get_db)):
    """Create a new city."""
    # Determine safety zone based on crime index
    if city.crime_index <= 30:
        safety_zone = "green"
    elif city.crime_index <= 60:
        safety_zone = "orange"
    else:
        safety_zone = "red"
    
    db_city = models.City(
        name=city.name,
        state=city.state,
        latitude=city.latitude,
        longitude=city.longitude,
        population=city.population,
        crime_index=city.crime_index,
        safety_zone=safety_zone
    )
    db.add(db_city)
    db.commit()
    db.refresh(db_city)
    return db_city
