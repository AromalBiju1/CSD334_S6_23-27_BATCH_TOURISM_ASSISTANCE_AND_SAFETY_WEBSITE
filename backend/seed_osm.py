"""
Script to seed database with real attractions across India using OpenStreetMap Overpass API and Wikipedia.
Queries predefined categories for every city found in the database.
"""
import sys, os, json, urllib.request, urllib.parse, urllib.error
import time, random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.database import sessionLocal
from database.models import Attraction, City

CATEGORIES = {
    'Monument': 'historic=monument',
    'Temple': 'amenity=place_of_worship][religion=hindu',
    'Palace': 'historic=castle][castle_type=palace',
    'Fort': 'historic=fort',
    'Beach': 'natural=beach',
    'Museum': 'tourism=museum',
    'Park': 'leisure=park'
}
LIMIT_PER_CATEGORY_PER_CITY = 3

def get_wikipedia_thumbnail(title):
    try:
        if not title:
            return None
        # Format title properly for Wikipedia API
        formatted_title = urllib.parse.quote(title.replace(" ", "_"))
        url = f"https://en.wikipedia.org/w/api.php?action=query&titles={formatted_title}&prop=pageimages&format=json&pithumbsize=500"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'TouristApp/1.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                if 'thumbnail' in page_data:
                    return page_data['thumbnail']['source']
    except Exception as e:
        pass
    return None

def fetch_osm_data(city_name, category, tag):
    query = f"""
    [out:json][timeout:25];
    area[name="{city_name}"]->.searchArea;
    node[{tag}](area.searchArea);
    out {LIMIT_PER_CATEGORY_PER_CITY};
    way[{tag}](area.searchArea);
    out center {LIMIT_PER_CATEGORY_PER_CITY};
    """
    
    url = "https://overpass-api.de/api/interpreter"
    data = query.encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'User-Agent': 'TouristApp/1.0'})
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            return result.get("elements", [])
    except Exception as e:
        print(f"    ⚠️  Error reaching OSM for {category} in {city_name}: {e}")
        return []

def run_seeder():
    db = sessionLocal()
    try:
        cities = db.query(City).all()
        print(f"Found {len(cities)} cities in database.")
        
        total_added = 0
        
        for city in cities:
            print(f"\nProcessing City: {city.name}, {city.state}")
            
            for category, tag in CATEGORIES.items():
                results = fetch_osm_data(city.name, category, tag)
                
                added_for_cat = 0
                for place in results:
                    tags = place.get('tags', {})
                    name = tags.get('name') or tags.get('name:en')
                    if not name:
                        continue
                        
                    # Skip if already exists
                    exists = db.query(Attraction).filter(Attraction.name == name, Attraction.city_id == city.id).first()
                    if exists:
                        continue
                        
                    # Extract coordinates (handles both node and way elements)
                    if place.get('type') == 'node':
                        lat = place.get('lat')
                        lng = place.get('lon')
                    elif place.get('type') == 'way' and 'center' in place:
                        lat = place.get('center', {}).get('lat')
                        lng = place.get('center', {}).get('lon')
                    else:
                        continue
                        
                    # Try to fetch Wikipedia image if linked in OSM
                    image_url = None
                    wiki_tag = tags.get('wikipedia')
                    if wiki_tag and ':' in wiki_tag:
                        # usually format is "en:Article Name"
                        lang, title = wiki_tag.split(':', 1)
                        if lang == 'en':
                            image_url = get_wikipedia_thumbnail(title)
                    
                    # If no OSM wikipedia tag, guess it using the name
                    if not image_url:
                        image_url = get_wikipedia_thumbnail(name)
                        
                    # Add dummy rating and description
                    rating = round(random.uniform(3.8, 4.9), 1)
                    desc = tags.get('description') or tags.get('description:en')
                    
                    if not desc:
                        website = tags.get('website')
                        desc = f"Attraction in {city.name}."
                        if website:
                            desc += f" More info at: {website}"
                            
                    if desc and len(desc) > 500:
                        desc = desc[:497] + "..."
                        
                    new_attr = Attraction(
                        city_id=city.id,
                        name=name,
                        category=category,
                        rating=rating,
                        latitude=lat,
                        longitude=lng,
                        description=desc,
                        image_url=image_url
                    )
                    db.add(new_attr)
                    added_for_cat += 1
                    total_added += 1
                    
                if added_for_cat > 0:
                    print(f"  ✓ Added {added_for_cat} {category}s")
                    
                # Respect Overpass API rate limits
                time.sleep(1)
                
            db.commit() 
            
        print(f"\n✅ Finished! Successfully added {total_added} new OSM attractions.")
        
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    run_seeder()
