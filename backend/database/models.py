from sqlalchemy import Column,Integer,String,Float, DateTime,Boolean,ForeignKey,Index
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database.database import Base
from datetime import datetime, timedelta

class User(Base):
    __tablename__= "users"
    id = Column(Integer, primary_key=True, index=True)
    password = Column(String, nullable=False) 
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, index=True, nullable=False)
    phone = Column(String, unique=True, nullable=True)
    profile_pic = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    google_id = Column(String, unique=True, nullable=True)
    
    # --- New Preference/Privacy Columns ---
    language = Column(String, default="English")
    theme = Column(String, default="light")
    notifications_enabled = Column(Boolean, default=True)
    is_public = Column(Boolean, default=True) # Privacy setting
    
    # Relationships
    saved_routes = relationship("SavedRoute", back_populates="user")
    preferences = relationship("UserPreference", back_populates="user", uselist=False)
    visited_places = relationship("VisitedPlace", back_populates="user")
    activities = relationship("ActivityHistory", back_populates="user")

class City(Base):
    __tablename__ = "cities"
    __table_args__ = (
        Index('ix_city_zone_state', 'safety_zone', 'state'),
    )
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    state = Column(String(100), nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    population = Column(Integer)
    crime_index = Column(Float, default=0.0) 
    safety_zone = Column(String(10), default="orange")
    crime_statistics = relationship("CrimeStatistic", back_populates="city")
    attractions = relationship("Attraction", back_populates="city") 

class CrimeStatistic(Base):
    __tablename__ = "crime_statistics" 
    id = Column(Integer, primary_key=True, index=True)
    city_id = Column(Integer, ForeignKey("cities.id"))
    year = Column(Integer)
    crime_rate = Column(Float)
    city = relationship("City", back_populates="crime_statistics")



class Attraction(Base):
    __tablename__ = "attractions"
    
    id = Column(Integer, primary_key=True, index=True)
    city_id = Column(Integer, ForeignKey("cities.id")) 
    name = Column(String(200), nullable=False)
    category = Column(String(50))
    rating = Column(Float)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    description = Column(String(500), nullable=True)
    image_url = Column(String(500), nullable=True)
    city = relationship("City", back_populates="attractions")  

class EmergencyContact(Base):
    __tablename__ = "emergency_contacts"
    id = Column(Integer, primary_key=True)
    city_id = Column(Integer, ForeignKey("cities.id"), nullable=True)
    name = Column(String)
    number = Column(String)
    service_type = Column(String) 
    is_national = Column(Boolean, default=False)

class SavedRoute(Base):
    __tablename__ = "saved_routes"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    origin = Column(String, nullable=False)
    destination = Column(String, nullable=False)
    distance_km = Column(Float, nullable=False)
    safety_score = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="saved_routes")


class UserPreference(Base):
    __tablename__ = "user_preferences"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    preferred_categories = Column(String, default="")  # comma-separated: "Temple,Beach,Museum"
    budget_level = Column(String, default="medium")  # low, medium, high
    travel_style = Column(String, default="balanced")  # adventure, relaxed, balanced
    preferred_safety = Column(String, default="all")  # green, orange, all
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    user = relationship("User", back_populates="preferences")


class VisitedPlace(Base):
    __tablename__ = "visited_places"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    attraction_id = Column(Integer, ForeignKey("attractions.id"))
    rating = Column(Float, nullable=True)  # user rating 1-5
    visited_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="visited_places")
    attraction = relationship("Attraction")

class ActivityHistory(Base):
    __tablename__ = "activity_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action_type = Column(String, nullable=False) # e.g., "explore", "plan", "hotspots"
    title = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="activities")