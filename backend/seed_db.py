"""
Seed Cities from Training Dataset
Populates the cities table with data from the ML training dataset.

Run from backend directory with venv activated:
    python seed_db.py
"""

import sys
import os
from pathlib import Path
import csv

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
TOURIST_DATA_PATH = SCRIPT_DIR.parent / "frontend" / "crime_data" / "india_tourist_destinations_full.csv"

# Import from database package
from database.database import sessionLocal, engine, Base
from database.models import City, EmergencyContact


def seed_cities_from_training_data():
    """Seed cities from tourist destinations CSV."""
    db = sessionLocal()
    
    try:
        # Clear existing cities
        db.query(City).delete()
        db.commit()
        print("Cleared existing cities")
        
        cities_added = 0
        seen_cities = set()
        
        if not TOURIST_DATA_PATH.exists():
            print(f"ERROR: Tourist data not found at {TOURIST_DATA_PATH}")
            return 0
        
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
                    crime_index = 25.0
                elif risk_label == 'Moderate':
                    safety_zone = 'orange'
                    crime_index = 50.0
                else:
                    safety_zone = 'red'
                    crime_index = 80.0
                
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
    db = sessionLocal()
    
    try:
        db.query(EmergencyContact).delete()
        db.commit()
        
        contacts = [
            ("Police", "100", "police", True),
            ("Ambulance", "102", "medical", True),
            ("Fire", "101", "fire", True),
            ("Women Helpline", "1091", "women", True),
            ("Tourist Helpline", "1363", "tourist", True),
            ("Emergency Response", "112", "emergency", True),
            ("Child Helpline", "1098", "child", True),
            ("Road Accident", "1073", "road", True),
        ]
        
        for name, number, service_type, is_national in contacts:
            db.add(EmergencyContact(
                name=name,
                number=number,
                service_type=service_type,
                is_national=is_national,
                city_id=None
            ))
        
        db.commit()
        print(f"Added {len(contacts)} emergency contacts")
        
    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    print("=" * 50)
    print("Seeding Database")
    print("=" * 50)
    
    Base.metadata.create_all(bind=engine)
    
    cities = seed_cities_from_training_data()
    seed_emergency_contacts()
    
    print("=" * 50)
    print(f"Done! {cities} cities added.")
    print("=" * 50)
