from sqlalchemy import Column,Integer,String,Float, DateTime,Boolean,ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database.database import Base
from datetime import datetime, timedelta

class User(Base):
    __tablename__= "users"
    id = Column(Integer, primary_key=True,index=True)
    password = Column(String, nullable=False) 
    email = Column(String, unique=True,index=True,nullable=False)
    name = Column(String, index=True,nullable=False)
    profile_pic = Column(String,nullable=True)
    google_id = Column(String,nullable=True,unique=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())



class City(Base):
    __tablename__ = "cities"
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
    city = relationship("City", back_populates="attractions")  

class EmergencyContact(Base):
    __tablename__ = "emergency_contacts"
    id = Column(Integer, primary_key=True)
    city_id = Column(Integer, ForeignKey("cities.id"), nullable=True)
    name = Column(String)
    number = Column(String)
    service_type = Column(String) .
    is_national = Column(Boolean, default=False)