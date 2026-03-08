"""
Seed script for Attractions using OpenTripMap API.
Fetches high-quality tourist attractions and their images.
"""
import sys
import os
import requests
import time
from dotenv import load_dotenv

# Ensure we can import from database
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import sessionLocal
from database.models import Attraction, City

load_dotenv()
OTM_API_KEY = os.getenv("OPENTRIPMAP_API")

if not OTM_API_KEY:
    print("❌ ERROR: OPENTRIPMAP_API key not found in environment.")
    sys.exit(1)

# Mapping OTM categories to our simpler structure
CATEGORY_MAP = {
    'historic': 'Monuments',
    'archaeology': 'Monuments',
    'monuments': 'Monuments',
    'museums': 'Museums',
    'temples': 'Temples',
    'hindu_temples': 'Temples',
    'mosques': 'Mosques',
    'churches': 'Churches',
    'natural': 'Parks/Nature',
    'nature_reserves': 'Parks/Nature',
    'beaches': 'Beaches',
    'amusement_parks': 'Amusement Parks',
    'forts': 'Forts',
    'castles': 'Forts',
    'places_of_worship': 'Temples'
}

def map_category(kinds: str) -> str:
    """Map OTM kinds to a simplified category"""
    kind_list = kinds.split(',')
    
    # Check specific matches first
    for kind in kind_list:
        if kind in CATEGORY_MAP:
            return CATEGORY_MAP[kind]
            
    # Fallback to general categories based on keywords
    if 'religion' in kinds or 'worship' in kinds:
        return 'Places of Worship'
    if 'historic' in kinds:
        return 'Historical'
    if 'cultural' in kinds:
        return 'Cultural'
    if 'natural' in kinds:
        return 'Nature'
        
    return 'Attraction'


def get_places_in_radius(lat: float, lon: float, radius: int = 15000, limit: int = 20):
    """Get places of interest near a coordinate"""
    url = f"https://api.opentripmap.com/0.1/en/places/radius"
    params = {
        'radius': radius,
        'lon': lon,
        'lat': lat,
        'kinds': 'tourist_facilities,interesting_places,cultural',
        'rate': '2', # Top and mid-tier rated places to ensure we don't miss obvious landmarks
        'format': 'json',
        'limit': limit,
        'apikey': OTM_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching from /radius: {e}")
        return []


def get_place_details(xid: str):
    """Get detailed info including image URL for a place"""
    url = f"https://api.opentripmap.com/0.1/en/places/xid/{xid}"
    params = {'apikey': OTM_API_KEY}
    
    try:
        # Rate limit protection (OTM allows 10/sec, this ensures we stay well under)
        time.sleep(0.2)
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching details for {xid}: {e}")
        return None


def run_seeder():
    # Fetch all cities first and detach them
    db = sessionLocal()
    try:
        cities_raw = db.query(City).all()
        city_data = [
            {"id": c.id, "name": c.name, "state": c.state, "latitude": c.latitude, "longitude": c.longitude}
            for c in cities_raw
        ]
    finally:
        db.close()
        
    # Move Thiruvananthapuram to the front to immediately satisfy the user
    trivandrum = next((c for c in city_data if c['name'].lower() in ['thiruvananthapuram', 'trivandrum']), None)
    if trivandrum:
        city_data.remove(trivandrum)
        city_data.insert(0, trivandrum)
        
    print(f"Found {len(city_data)} cities to process.")
    
    total_added = 0
    total_processed = 0
    
    for city in city_data:
        # Skip if city doesn't have coordinates
        if not city['latitude'] or not city['longitude']:
            continue
            
        print(f"\n📍 Fetching attractions for {city['name']}, {city['state']}...")
        
        # 1. Find places in radius
        places = get_places_in_radius(city['latitude'], city['longitude'], limit=50) # Fetch more candidates
        
        if not places:
            print(f"  No highly-rated places found.")
            continue
            
        added_for_city = 0
        
        # Open a fresh session for each city to avoid long-running transaction timeouts
        db = sessionLocal()
        try:
            # 2. Get details for each place
            for place in places:
                total_processed += 1
                name = place.get('name')
                xid = place.get('xid')
                
                if not name or not name.strip():
                    continue
                    
                # Sanitize common bad names (often coming from generic nodes in OSM)
                lower_name = name.strip().lower()
                if lower_name in ['temple', 'hindu temple', 'temble', 'church', 'mosque', 'park', 'beach', 'monument', 'museum']:
                    continue
                    
                # Check if it already exists
                exists = db.query(Attraction).filter(
                    Attraction.name == name,
                    Attraction.city_id == city['id']
                ).first()
                
                if exists:
                    continue
                    
                # Get detailed info to grab the image!
                details = get_place_details(xid)
                if not details:
                    continue
                    
                # Extract image and description
                image_url = None
                if 'preview' in details and 'source' in details['preview']:
                    image_url = details['preview']['source']
                    
                # Fallback to wiki if OpenTripMap doesn't have a direct preview
                if not image_url and 'wikipedia' in details:
                    wiki_link = details['wikipedia']
                    # We could do a secondary call here, but let's stick to what OTM provides for now
                    
                description = None
                if 'wikipedia_extracts' in details and 'text' in details['wikipedia_extracts']:
                    description = details['wikipedia_extracts']['text']
                elif 'info' in details and 'descr' in details['info']:
                    description = details['info']['descr']
                    
                # Truncate description if too long
                if description and len(description) > 500:
                    description = description[:497] + "..."
                    
                # Truncate names and urls if too long
                if image_url and len(image_url) > 255:
                    image_url = image_url[:255]
                if name and len(name) > 100:
                    name = name[:97] + "..."
                    
                # Map category from comma-separated kinds string
                kinds = details.get('kinds', '')
                category = map_category(kinds)
                
                # Default rating since OTM rate is a scale of 1-3, we'll map to 3.5 - 5.0
                otm_rate = str(details.get('rate', '1')).replace('h', '')
                try:
                    rating = 4.0 + (int(otm_rate) * 0.3)
                except ValueError:
                    rating = 4.5
                if rating > 5.0: rating = 4.9
                
                attr = Attraction(
                    name=name,
                    category=category,
                    rating=round(rating, 1),
                    latitude=details.get('point', {}).get('lat', place.get('point', {}).get('lat')),
                    longitude=details.get('point', {}).get('lon', place.get('point', {}).get('lon')),
                    description=description,
                    image_url=image_url,
                    city_id=city['id']
                )
                
                db.add(attr)
                added_for_city += 1
                total_added += 1
                
                if added_for_city >= 15:  # Let's limit to 15 attractions per city instead of 5
                    break
                    
            if added_for_city > 0:
                print(f"  ✓ Added {added_for_city} attractions.")
                db.commit()  # Commit per city so we save progress
                
        except Exception as e:
            print(f"❌ Error processing {city['name']}: {e}")
            db.rollback()
        finally:
            db.close()
            
    print(f"\n✅ Finished! Processed {total_processed} places and added {total_added} new detailed attractions.")

if __name__ == "__main__":
    run_seeder()
