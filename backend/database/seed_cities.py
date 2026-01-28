"""
Seed Cities from Training Dataset
Populates the cities table with data from the ML training dataset.

Run from backend directory: python -m database.seed_cities
Or move to backend folder: python seed_cities.py
"""

import sys
import os
from pathlib import Path
import csv

# Setup path - add backend directory to sys.path
SCRIPT_DIR = Path(__file__).parent.absolute()
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

# Now import using relative imports
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config.config import database_url

# Create engine and session directly to avoid import issues
engine = create_engine(database_url)
SessionLocal = sessionmaker(autoflush=False, autocommit=False, bind=engine)
Base = declarative_base()

# Import models after setting up path
from database.models import City, EmergencyContact, Base


# Data paths
TOURIST_DATA_PATH = BACKEND_DIR.parent / "frontend" / "crime_data" / "india_tourist_destinations_full.csv"
TRAINING_DATA_PATH = BACKEND_DIR.parent / "frontend" / "ml" / "training_dataset.csv"


def determine_safety_zone(risk_label: int) -> str:
    """Map risk label to safety zone."""
    if risk_label == 0:
        return "green"
    elif risk_label == 1:
        return "orange"
    else:
        return "red"


def seed_cities_from_training_data():
    """Seed cities from ML training dataset."""
    db = SessionLocal()
    
    try:
        # Clear existing cities
        db.query(City).delete()
        db.commit()
        print("Cleared existing cities")
        
        cities_added = 0
        seen_cities = set()
        
        # Try tourist destinations first (has coordinates)
        if TOURIST_DATA_PATH.exists():
            print(f"Loading from {TOURIST_DATA_PATH}")
            with open(TOURIST_DATA_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row.get('Location'):
                        continue
                    
                    name = row['Location'].replace('_', ' ')
                    state = row.get('State', '').replace('_', ' ')
                    key = f"{name}_{state}".lower()
                    
                    if key in seen_cities:
                        continue
                    seen_cities.add(key)
                    
                    # Map risk label
                    risk_label = row.get('Risk_Label', 'Moderate')
                    if risk_label == 'Safe':
                        safety_zone = 'green'
                        crime_index = 25
                    elif risk_label == 'Moderate':
                        safety_zone = 'orange'
                        crime_index = 50
                    else:
                        safety_zone = 'red'
                        crime_index = 80
                    
                    try:
                        lat = float(row.get('Latitude', 0))
                        lng = float(row.get('Longitude', 0))
                    except (ValueError, TypeError):
                        lat, lng = 0, 0
                    
                    if lat == 0 or lng == 0:
                        continue
                    
                    city = City(
                        name=name,
                        state=state,
                        latitude=lat,
                        longitude=lng,
                        population=None,
                        crime_index=crime_index,
                        safety_zone=safety_zone
                    )
                    db.add(city)
                    cities_added += 1
            
            db.commit()
            print(f"Added {cities_added} cities from tourist destinations")
        else:
            print(f"Tourist data not found at {TOURIST_DATA_PATH}")
        
        return cities_added
        
    except Exception as e:
        db.rollback()
        print(f"Error seeding cities: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        db.close()


def seed_emergency_contacts():
    """Seed national emergency contacts."""
    db = SessionLocal()
    
    try:
        # Clear existing
        db.query(EmergencyContact).delete()
        db.commit()
        
        national_contacts = [
            ("Police", "100", "police", True),
            ("Ambulance", "102", "medical", True),
            ("Fire", "101", "fire", True),
            ("Women Helpline", "1091", "women", True),
            ("Tourist Helpline", "1363", "tourist", True),
            ("Emergency Response", "112", "emergency", True),
            ("Child Helpline", "1098", "child", True),
            ("Road Accident Emergency", "1073", "road", True),
        ]
        
        for name, number, service_type, is_national in national_contacts:
            contact = EmergencyContact(
                name=name,
                number=number,
                service_type=service_type,
                is_national=is_national,
                city_id=None
            )
            db.add(contact)
        
        db.commit()
        print(f"Added {len(national_contacts)} emergency contacts")
        
    except Exception as e:
        db.rollback()
        print(f"Error seeding emergency contacts: {e}")
        raise
    finally:
        db.close()


def main():
    """Run all seed functions."""
    print("=" * 50)
    print("Seeding Database")
    print("=" * 50)
    print(f"Backend dir: {BACKEND_DIR}")
    print(f"Tourist data: {TOURIST_DATA_PATH}")
    
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    cities_count = seed_cities_from_training_data()
    seed_emergency_contacts()
    
    print("=" * 50)
    print(f"Seeding complete! {cities_count} cities added.")
    print("=" * 50)


if __name__ == "__main__":
    main()
