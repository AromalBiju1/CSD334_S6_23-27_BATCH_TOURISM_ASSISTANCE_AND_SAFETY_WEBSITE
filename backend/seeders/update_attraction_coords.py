"""
One-time script to populate real-world GPS coordinates for all seeded attractions.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.database import sessionLocal
from database.models import Attraction

# Accurate real-world coordinates for every seeded attraction
COORDS = {
    # Trivandrum
    "Padmanabhaswamy Temple":   (8.4826, 76.9438),
    "Kovalam Beach":            (8.3988, 76.9783),
    "Napier Museum":            (8.5116, 76.9554),
    "Villa Maya":               (8.5110, 76.9590),

    # Udaipur
    "City Palace":              (24.5764, 73.6913),
    "Lake Pichola":             (24.5710, 73.6808),
    "Kumbhalgarh Fort":         (25.1520, 73.5870),
    "Ambrai Restaurant":        (24.5725, 73.6790),
    "Saheliyon-ki-Bari":        (24.5912, 73.6980),

    # Varanasi
    "Kashi Vishwanath Temple":  (25.3109, 83.0107),
    "Sarnath":                  (25.3814, 83.0226),
    "Ramnagar Fort":            (25.2875, 83.0283),
    "Shree Shivay Thali":       (25.3176, 83.0104),

    # Visakhapatnam
    "Rishi Konda Beach":        (17.7897, 83.3843),
    "INS Kursura":              (17.7146, 83.3237),
    "Kailasagiri":              (17.7570, 83.3716),

    # Alappuzha
    "Alleppey Beach":           (9.4900, 76.3264),
    "Marari Beach":             (9.6187, 76.3060),
    "Thaff Restaurant":         (9.4935, 76.3375),

    # Vellore
    "Golden Temple":            (12.9249, 79.1325),
    "Vellore Fort":             (12.9165, 79.1325),
}

def update_coords():
    db = sessionLocal()
    try:
        updated = 0
        for name, (lat, lng) in COORDS.items():
            attr = db.query(Attraction).filter(Attraction.name == name).first()
            if attr:
                attr.latitude = lat
                attr.longitude = lng
                updated += 1
                print(f"  ✓ {name} -> ({lat}, {lng})")
            else:
                print(f"  ✗ Not found: {name}")
        db.commit()
        print(f"\n✅ Updated {updated}/{len(COORDS)} attractions with coordinates.")
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    update_coords()
