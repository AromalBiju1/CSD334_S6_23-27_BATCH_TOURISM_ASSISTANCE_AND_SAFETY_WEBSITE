"""
Script to seed database with real attractions across India using Foursquare Places API v3.
Queries predefined categories for every city found in the database.
"""
import sys, os, json, urllib.request, urllib.parse, urllib.error
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from database.database import sessionLocal
from database.models import Attraction, City

load_dotenv()
FSQ_API_KEY = os.getenv("FOURSQUARE_API_KEY", "").strip('"').strip("'")

CATEGORIES = ['Monument', 'Temple', 'Palace', 'Fort', 'Beach', 'Museum', 'Park']
LIMIT_PER_CATEGORY_PER_CITY = 3

def fetch_foursquare_data(city_name, state_name, category):
    # Foursquare Places API v3
    # We ask for specific fields to include rating and description
    fields = "fsq_id,name,location,categories,geocodes,rating,description"
    
    # We pass near the city and state to narrow down.
    near = f"{city_name}, {state_name}, India"
    url = f"https://api.foursquare.com/v3/places/search?near={urllib.parse.quote(near)}&query={urllib.parse.quote(category)}&limit={LIMIT_PER_CATEGORY_PER_CITY}&fields={fields}"
    
    headers = {
        "Authorization": FSQ_API_KEY,
        "accept": "application/json"
    }
    
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data.get("results", [])
    except urllib.error.HTTPError as e:
        # 400 Bad Request usually means 'near' could not be resolved.
        if e.code != 400:
            print(f"    ⚠️  Foursquare API error for {category} in {city_name}: {e.code}")
        return []
    except Exception as e:
        print(f"    ⚠️  Error reaching Foursquare for {category} in {city_name}: {e}")
        return []

def run_seeder():
    if not FSQ_API_KEY:
        print("❌ Error: FOURSQUARE_API_KEY not found in .env")
        return

    db = sessionLocal()
    try:
        cities = db.query(City).all()
        print(f"Found {len(cities)} cities in database.")
        
        total_added = 0
        
        for city in cities:
            print(f"Processing City: {city.name}, {city.state}")
            
            for category in CATEGORIES:
                results = fetch_foursquare_data(city.name, city.state, category)
                
                added_for_cat = 0
                for place in results:
                    name = place.get("name")
                    if not name:
                        continue
                        
                    # Skip if already exists
                    exists = db.query(Attraction).filter(Attraction.name == name, Attraction.city_id == city.id).first()
                    if exists:
                        continue
                        
                    lat = place.get("geocodes", {}).get("main", {}).get("latitude")
                    lng = place.get("geocodes", {}).get("main", {}).get("longitude")
                    
                    # Ratings are out of 10 in Foursquare, convert to 5 star
                    fsq_rating = place.get("rating")
                    if fsq_rating:
                        rating = round(fsq_rating / 2.0, 1)
                    else:
                        rating = round(random.uniform(3.8, 4.9), 1)
                        
                    # Format description or use address
                    desc = place.get("description")
                    if not desc:
                        address = place.get("location", {}).get("formatted_address")
                        desc = f"Located at: {address}" if address else None
                        
                    # Constrain length
                    if desc and len(desc) > 500:
                        desc = desc[:497] + "..."
                        
                    new_attr = Attraction(
                        city_id=city.id,
                        name=name,
                        category=category,
                        rating=rating,
                        latitude=lat,
                        longitude=lng,
                        description=desc
                    )
                    db.add(new_attr)
                    added_for_cat += 1
                    total_added += 1
                    
                if added_for_cat > 0:
                    print(f"  ✓ Added {added_for_cat} {category}s")
                    
            db.commit()  # Commit per city so we keep progress
            
        print(f"\n✅ Finished! Successfully added {total_added} new attractions.")
        
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    run_seeder()
