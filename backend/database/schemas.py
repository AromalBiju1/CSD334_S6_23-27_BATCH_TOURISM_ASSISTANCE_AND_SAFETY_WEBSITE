from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime
from enum import Enum

class SafetyZone(str, Enum):
    GREEN = "green"
    ORANGE = "orange"
    RED = "red"

class UserBase(BaseModel):
    email: EmailStr
    name: str

class UserCreate(UserBase):
    google_id: Optional[str] = None

class UserResponse(UserBase):
    id: int
    profile_pic: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True
# C I T Y
class CityBase(BaseModel):
    name: str = Field(..., max_length=100)
    state: str = Field(..., max_length=100)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

class CityCreate(CityBase):
    population: Optional[int] = None
    crime_index: float = 50.0

class CityResponse(CityBase):
    id: int
    population: Optional[int]
    crime_index: float
    safety_zone: str
    
    class Config:
        from_attributes = True


# CRIME 
class CrimeStatisticBase(BaseModel):
    year: int
    crime_rate: float

class CrimeStatisticCreate(CrimeStatisticBase):
    city_id: int

class CrimeStatisticResponse(CrimeStatisticBase):
    id: int
    city_id: int
    
    class Config:
        from_attributes = True


# ATTRACTION 
class AttractionBase(BaseModel):
    name: str = Field(..., max_length=200)
    category: Optional[str] = None
    rating: Optional[float] = Field(None, ge=0, le=5)

class AttractionCreate(AttractionBase):
    city_id: int

class AttractionResponse(AttractionBase):
    id: int
    city_id: int
    
    class Config:
        from_attributes = True

class CityWithDetails(CityResponse):
    """City with related data"""
    attractions: List[AttractionResponse] = []
    crime_statistics: List[CrimeStatisticResponse] = []