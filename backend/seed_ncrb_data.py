"""
Seed database with NCRB District Crime Data
This script parses NCRB_District_Table_1.3.csv and populates the cities table
with proper crime statistics and safety zone classifications.
"""
import os
import csv
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# District coordinates (major districts - will be geocoded for others)
DISTRICT_COORDS = {
    # Major metro cities
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Ahmedabad": (23.0225, 72.5714),
    "Pune": (18.5204, 73.8567),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
    "Nagpur": (21.1458, 79.0882),
    "Indore": (22.7196, 75.8577),
    "Thane": (19.2183, 72.9781),
    "Bhopal": (23.2599, 77.4126),
    "Patna": (25.5941, 85.1376),
    "Vadodara": (22.3072, 73.1812),
    "Surat": (21.1702, 72.8311),
    "Kochi": (9.9312, 76.2673),
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Coimbatore": (11.0168, 76.9558),
    "Visakhapatnam": (17.6868, 83.2185),
    "Agra": (27.1767, 78.0081),
    "Varanasi": (25.3176, 82.9739),
    "Guwahati": (26.1445, 91.7362),
    "Chandigarh": (30.7333, 76.7794),
    "Amritsar": (31.6340, 74.8723),
    "Goa": (15.2993, 74.1240),
    "Shimla": (31.1048, 77.1734),
    "Dehradun": (30.3165, 78.0322),
    "Ranchi": (23.3441, 85.3096),
    "Raipur": (21.2514, 81.6296),
    "Bhubaneswar": (20.2961, 85.8245),
    # Kerala districts
    "Ernakulam": (9.9816, 76.2999),
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Kozhikode": (11.2588, 75.7804),
    "Thrissur": (10.5276, 76.2144),
    "Kollam": (8.8932, 76.6141),
    "Palakkad": (10.7867, 76.6548),
    "Kannur": (11.8745, 75.3704),
    "Malappuram": (11.0510, 76.0711),
    "Alappuzha": (9.4981, 76.3388),
    "Idukki": (9.8494, 77.1025),
    "Kasaragod": (12.4996, 74.9869),
    "Wayanad": (11.6854, 76.1320),
    "Pathanamthitta": (9.2648, 76.7870),
    "Kottayam": (9.5916, 76.5222),
    # Add more as needed
}

# State center coordinates for fallback
STATE_CENTERS = {
    "Andhra Pradesh": (15.9129, 79.7400),
    "Arunachal Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chhattisgarh": (21.2787, 81.8661),
    "Goa": (15.2993, 74.1240),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Sikkim": (27.5330, 88.5122),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Tripura": (23.9408, 91.9882),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West Bengal": (22.9868, 87.8550),
    "Delhi": (28.7041, 77.1025),
}

def get_coordinates(district, state):
    """Get coordinates for a district, with fallbacks"""
    # Try exact district match
    for key, coords in DISTRICT_COORDS.items():
        if key.lower() in district.lower() or district.lower() in key.lower():
            return coords
    
    # Try state center as fallback with small offset
    if state in STATE_CENTERS:
        lat, lng = STATE_CENTERS[state]
        # Add small random offset to avoid overlapping
        import random
        offset_lat = random.uniform(-0.5, 0.5)
        offset_lng = random.uniform(-0.5, 0.5)
        return (lat + offset_lat, lng + offset_lng)
    
    return None

def calculate_crime_index(row):
    """Calculate crime index from NCRB data columns"""
    try:
        # Key crime indicators (with column indices from the CSV)
        # Col 54 = Total Crime against Women
        total_crimes = float(row[53]) if row[53] else 0
        
        # Individual crime types weighted by severity
        murder_rape = float(row[3]) if row[3] else 0  # Murder with Rape
        dowry_deaths = float(row[4]) if row[4] else 0
        rape = float(row[22]) if row[22] else 0  # Total Rape
        kidnapping = float(row[10]) if row[10] else 0  # K&A Total
        assault = float(row[28]) if row[28] else 0  # Assault on women
        
        # Weighted crime score
        weighted_score = (
            murder_rape * 10 +
            dowry_deaths * 8 +
            rape * 6 +
            kidnapping * 4 +
            assault * 2 +
            total_crimes * 0.1
        )
        
        return weighted_score
    except (ValueError, IndexError):
        return 0

def classify_safety_zone(crime_index, percentile_25, percentile_75):
    """Classify district into safety zones based on crime index percentiles"""
    if crime_index <= percentile_25:
        return "green"  # Safe
    elif crime_index <= percentile_75:
        return "orange"  # Moderate
    else:
        return "red"  # High Risk

def main():
    # Read environment variables
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not found in .env")
        return
    
    # Read CSV file
    csv_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "crime_data", "NCRB_District_Table_1.3.csv")
    
    districts = []
    
    print("📊 Reading NCRB crime data...")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 54:
                continue
            
            # Skip summary rows
            if "Total Districts" in row[0] or "Total Districts" in row[2]:
                continue
            
            # Skip non-district entries (like CID, GRP, Railways, etc.)
            skip_keywords = ['CID', 'GRP', 'Railway', 'Cyber', 'STF', 'SOU', 'EOW', 
                           'Anti Terrorist', 'Narcotic', 'Intelligence', 'Bureau',
                           'Special Task', 'Crime Branch', 'Other Units']
            district_name = row[2].strip()
            if any(kw.lower() in district_name.lower() for kw in skip_keywords):
                continue
            
            state = row[1].strip()
            crime_index = calculate_crime_index(row)
            
            coords = get_coordinates(district_name, state)
            if not coords:
                continue
            
            districts.append({
                "name": district_name,
                "state": state,
                "latitude": coords[0],
                "longitude": coords[1],
                "crime_index": crime_index
            })
    
    print(f"✅ Parsed {len(districts)} valid districts")
    
    # Calculate percentiles for classification
    crime_indices = [d["crime_index"] for d in districts]
    crime_indices.sort()
    percentile_25 = crime_indices[int(len(crime_indices) * 0.25)]
    percentile_75 = crime_indices[int(len(crime_indices) * 0.75)]
    
    print(f"📈 Crime index percentiles: P25={percentile_25:.1f}, P75={percentile_75:.1f}")
    
    # Classify districts
    for d in districts:
        d["safety_zone"] = classify_safety_zone(d["crime_index"], percentile_25, percentile_75)
    
    # Count by zone
    zone_counts = {"green": 0, "orange": 0, "red": 0}
    for d in districts:
        zone_counts[d["safety_zone"]] += 1
    
    print(f"🟢 Safe: {zone_counts['green']} | 🟠 Moderate: {zone_counts['orange']} | 🔴 High Risk: {zone_counts['red']}")
    
    # Connect to database and insert
    print("\n🔄 Connecting to database...")
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()
    
    # Clear existing data
    cur.execute("DELETE FROM cities")
    print("🗑️ Cleared existing city data")
    
    # Insert new data
    inserted = 0
    for d in districts:
        try:
            cur.execute("""
                INSERT INTO cities (name, state, latitude, longitude, safety_zone, crime_index)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (d["name"], d["state"], d["latitude"], d["longitude"], d["safety_zone"], d["crime_index"]))
            inserted += 1
        except Exception as e:
            print(f"⚠️ Error inserting {d['name']}: {e}")
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"\n✅ Successfully inserted {inserted} districts into database!")
    print("\n📋 Sample entries:")
    for d in districts[:5]:
        print(f"   {d['name']}, {d['state']} - {d['safety_zone'].upper()} (index: {d['crime_index']:.1f})")

if __name__ == "__main__":
    main()
