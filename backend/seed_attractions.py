import sys
import os

# 1. Path setup: Ensures this script can see the 'database' folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import sessionLocal
from database.models import City, Attraction

# 2. THE HELPER FUNCTION (Must be defined before seed_abundance calls it)
def get_city_id(db, name):
    """
    Finds the city ID by name. 
    Uses 'ilike' and wildcards so 'Udaipur' matches 'Udaipur District'.
    """
    city = db.query(City).filter(City.name.ilike(f"%{name}%")).first()
    if city:
        return city.id
    return None

# 3. THE SEEDER FUNCTION
def seed_abundance():
    db = sessionLocal()
    try:
        # Your specific categories: Temples, Palaces, Forts, Beaches, Museums, Parks, Popular Restaurants
        data = {
            "Trivandrum": [
                {"name": "Padmanabhaswamy Temple", "category": "Temples", "rating": 4.9},
                {"name": "Kovalam Beach", "category": "Beaches", "rating": 4.7},
                {"name": "Napier Museum", "category": "Museums", "rating": 4.5},
                {"name": "Villa Maya", "category": "Popular Restaurants", "rating": 4.6}
            ],
            "Udaipur": [
                {"name": "City Palace", "category": "Palaces", "rating": 4.8},
                {"name": "Lake Pichola", "category": "Monuments", "rating": 4.9},
                {"name": "Kumbhalgarh Fort", "category": "Forts", "rating": 4.7},
                {"name": "Ambrai Restaurant", "category": "Popular Restaurants", "rating": 4.7},
                {"name": "Saheliyon-ki-Bari", "category": "Parks", "rating": 4.5}
            ],
            "Varanasi": [
                {"name": "Kashi Vishwanath Temple", "category": "Temples", "rating": 4.9},
                {"name": "Sarnath", "category": "Monuments", "rating": 4.7},
                {"name": "Ramnagar Fort", "category": "Forts", "rating": 4.3},
                {"name": "Shree Shivay Thali", "category": "Popular Restaurants", "rating": 4.5}
            ],
            "Visakhapatnam": [
                {"name": "Rishi Konda Beach", "category": "Beaches", "rating": 4.6},
                {"name": "INS Kursura", "category": "Museums", "rating": 4.8},
                {"name": "Kailasagiri", "category": "Parks", "rating": 4.5}
            ],
            "Alappuzha": [
                {"name": "Alleppey Beach", "category": "Beaches", "rating": 4.4},
                {"name": "Marari Beach", "category": "Beaches", "rating": 4.7},
                {"name": "Thaff Restaurant", "category": "Popular Restaurants", "rating": 4.2}
            ],
            "Vellore": [
                {"name": "Golden Temple", "category": "Temples", "rating": 4.8},
                {"name": "Vellore Fort", "category": "Forts", "rating": 4.5}
            ]
        }

        total_added = 0
        for city_name, attractions in data.items():
            # Calling the helper function defined above
            city_id = get_city_id(db, city_name)
            
            if city_id:
                for attr_data in attractions:
                    # Check for duplicates so you don't double-seed
                    exists = db.query(Attraction).filter(
                        Attraction.name == attr_data["name"],
                        Attraction.city_id == city_id
                    ).first()
                    
                    if not exists:
                        new_attr = Attraction(
                            name=attr_data["name"],
                            category=attr_data["category"],
                            rating=attr_data["rating"],
                            city_id=city_id
                        )
                        db.add(new_attr)
                        total_added += 1
            else:
                print(f"⚠️ City not found in database: {city_name}")

        db.commit()
        print(f"✅ Successfully seeded {total_added} attractions!")

    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

# 4. SCRIPT ENTRY POINT
if __name__ == "__main__":
    seed_abundance()