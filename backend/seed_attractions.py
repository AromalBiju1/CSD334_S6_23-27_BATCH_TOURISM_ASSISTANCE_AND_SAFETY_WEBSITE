from database.database import sessionLocal
from database.models import City, Attraction

def get_city_id(db, name):
    """Helper to find city ID even if the name in DB is 'Varanasi Commissionarate'"""
    city = db.query(City).filter(City.name.ilike(f"%{name}%")).first()
    return city.id if city else None

def seed_abundance():
    db = sessionLocal()
    try:
        # Dictionary of City Name -> List of Attractions
        data = {
            "Trivandrum": [
                {"name": "Padmanabhaswamy Temple", "category": "Religious", "rating": 4.9},
                {"name": "Kovalam Beach", "category": "Nature", "rating": 4.7},
                {"name": "Napier Museum", "category": "Museum", "rating": 4.5},
                {"name": "Shangumugham Beach", "category": "Nature", "rating": 4.4}
            ],
            "Udaipur": [
                {"name": "City Palace", "category": "Historical", "rating": 4.8},
                {"name": "Lake Pichola", "category": "Nature", "rating": 4.9},
                {"name": "Jagmandir", "category": "Historical", "rating": 4.6},
                {"name": "Saheliyon-ki-Bari", "category": "Park", "rating": 4.5}
            ],
            "Varanasi": [
                {"name": "Kashi Vishwanath Temple", "category": "Religious", "rating": 4.9},
                {"name": "Dashashwamedh Ghat", "category": "Culture", "rating": 4.8},
                {"name": "Sarnath", "category": "Historical", "rating": 4.7},
                {"name": "Assi Ghat", "category": "Culture", "rating": 4.6}
            ],
            "Visakhapatnam": [
                {"name": "Rishi Konda Beach", "category": "Nature", "rating": 4.6},
                {"name": "Kailasagiri", "category": "Park", "rating": 4.5},
                {"name": "INS Kursura Submarine Museum", "category": "Museum", "rating": 4.8}
            ],
            "Alappuzha": [
                {"name": "Alleppey Backwaters", "category": "Nature", "rating": 4.9},
                {"name": "Alappuzha Beach", "category": "Nature", "rating": 4.4},
                {"name": "Marari Beach", "category": "Nature", "rating": 4.7}
            ],
            "Thrissur": [
                {"name": "Vadakkunnathan Temple", "category": "Religious", "rating": 4.8},
                {"name": "Athirappilly Waterfalls", "category": "Nature", "rating": 4.9},
                {"name": "Thrissur Zoo", "category": "Nature", "rating": 4.2}
            ],
            "Ujjain": [
                {"name": "Mahakaleshwar Jyotirlinga", "category": "Religious", "rating": 4.9},
                {"name": "Kal Bhairav Temple", "category": "Religious", "rating": 4.7},
                {"name": "Ram Ghat", "category": "Culture", "rating": 4.5}
            ],
            "Thanjavur": [
                {"name": "Brihadisvara Temple", "category": "Historical", "rating": 4.9},
                {"name": "Thanjavur Maratha Palace", "category": "Historical", "rating": 4.4}
            ],
            "Tawang": [
                {"name": "Tawang Monastery", "category": "Religious", "rating": 4.9},
                {"name": "Sela Pass", "category": "Nature", "rating": 4.8},
                {"name": "Madhuri Lake", "category": "Nature", "rating": 4.7}
            ],
            "Vellore": [
                {"name": "Golden Temple (Sripuram)", "category": "Religious", "rating": 4.8},
                {"name": "Vellore Fort", "category": "Historical", "rating": 4.5}
            ]
        }

        total_added = 0
        for city_name, attractions in data.items():
            city_id = get_city_id(db, city_name)
            
            if city_id:
                for attr_data in attractions:
                    # Check for duplicates
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
                print(f"⚠️ Could not find city: {city_name} in database.")

        db.commit()
        print(f"✅ Successfully seeded {total_added} attractions!")

    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_abundance()