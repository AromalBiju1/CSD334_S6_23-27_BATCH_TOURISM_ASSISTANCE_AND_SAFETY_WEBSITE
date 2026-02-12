"""
Emergency Contacts Router
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel

from database.database import get_db
from database import models

router = APIRouter(prefix="/api/emergency", tags=["emergency"])


class EmergencyContactResponse(BaseModel):
    id: int
    name: str
    number: str
    service_type: str
    is_national: bool
    city_id: int | None = None
    
    class Config:
        from_attributes = True


@router.get("", response_model=List[EmergencyContactResponse])
def get_all_emergency_contacts(db: Session = Depends(get_db)):
    """Get all emergency contacts (national)."""
    contacts = db.query(models.EmergencyContact).filter(
        models.EmergencyContact.is_national == True
    ).all()
    return contacts


@router.get("/{city_id}", response_model=List[EmergencyContactResponse])
def get_city_emergency_contacts(city_id: int, db: Session = Depends(get_db)):
    """Get emergency contacts for a specific city."""
    # Get city-specific and national contacts
    contacts = db.query(models.EmergencyContact).filter(
        (models.EmergencyContact.city_id == city_id) |
        (models.EmergencyContact.is_national == True)
    ).all()
    
    if not contacts:
        # Return just national contacts if city has none
        contacts = db.query(models.EmergencyContact).filter(
            models.EmergencyContact.is_national == True
        ).all()
    
    return contacts
