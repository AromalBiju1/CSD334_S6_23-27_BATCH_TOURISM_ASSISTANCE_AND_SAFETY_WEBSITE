"""
Seed Database with XGBoost Predictions
This script uses the trained XGBoost model predictions to populate the cities table.
"""
import os
import csv
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_PATH = os.path.join(BASE_DIR, "..", "ml", "models", "district_xgboost_predictions.csv")

# District coordinates (comprehensive list)
DISTRICT_COORDS = {
    # Major metros
    "Mumbai": (19.0760, 72.8777),
    "Mumbai City": (19.0760, 72.8777),
    "Mumbai Suburban": (19.1136, 72.8697),
    "Delhi": (28.6139, 77.2090),
    "New Delhi": (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
    "Bangalore City": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Hyderabad City": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Ahmedabad": (23.0225, 72.5714),
    "Ahmedabad City": (23.0225, 72.5714),
    "Pune": (18.5204, 73.8567),
    "Pune City": (18.5204, 73.8567),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
    "Kanpur City": (26.4499, 80.3319),
    "Nagpur": (21.1458, 79.0882),
    "Indore": (22.7196, 75.8577),
    "Thane": (19.2183, 72.9781),
    "Bhopal": (23.2599, 77.4126),
    "Patna": (25.5941, 85.1376),
    "Vadodara": (22.3072, 73.1812),
    "Surat": (21.1702, 72.8311),
    "Surat City": (21.1702, 72.8311),
    
    # Kerala - expanded with City/Rural variants from 2024 data
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Thiruvananthapuram City": (8.5241, 76.9366),
    "Thiruvananthapuram Rural": (8.6500, 77.0500),
    "Ernakulam": (9.9816, 76.2999),
    "Ernakulam City": (9.9312, 76.2673),  # Kochi city
    "Ernakulam Rural": (10.0500, 76.3500),
    "Ernakulam Commr.": (9.9312, 76.2673),  # NCRB name for Kochi
    "Kochi": (9.9312, 76.2673),  # Display name
    "Kozhikode": (11.2588, 75.7804),
    "Kozhikode City": (11.2588, 75.7804),
    "Thrissur": (10.5276, 76.2144),
    "Thrissur City": (10.5276, 76.2144),
    "Kollam": (8.8932, 76.6141),
    "Kollam City": (8.8932, 76.6141),
    "Palakkad": (10.7867, 76.6548),
    "Kannur": (11.8745, 75.3704),
    "Kannur City": (11.8745, 75.3704),
    "Malappuram": (11.0510, 76.0711),
    "Alappuzha": (9.4981, 76.3388),
    "Alapuzha": (9.4981, 76.3388),
    "Idukki": (9.8494, 77.1025),
    "Kasaragod": (12.4996, 74.9869),
    "Kasargod": (12.4996, 74.9869),  # Alternative spelling
    "Wayanad": (11.6854, 76.1320),
    "Pathanamthitta": (9.2648, 76.7870),
    "Kottayam": (9.5916, 76.5222),
    
    # Andhra Pradesh
    "Visakhapatnam": (17.6868, 83.2185),
    "Vijayawada": (16.5062, 80.6480),
    "Guntur": (16.3067, 80.4365),
    "Tirupati": (13.6288, 79.4192),
    "Kurnool": (15.8281, 78.0373),
    "Nellore": (14.4426, 79.9865),
    "Anantapur": (14.6819, 77.6006),
    "Chittoor": (13.2172, 79.1003),
    
    # Tamil Nadu
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    "Tiruchirappalli": (10.7905, 78.7047),
    "Salem": (11.6643, 78.1460),
    "Tirunelveli": (8.7139, 77.7567),
    "Erode": (11.3410, 77.7172),
    "Vellore": (12.9165, 79.1325),
    "Thanjavur": (10.7870, 79.1378),
    
    # Karnataka
    "Mysuru": (12.2958, 76.6394),
    "Mysore": (12.2958, 76.6394),
    "Mangalore": (12.9141, 74.8560),
    "Hubli-Dharwad": (15.3647, 75.1240),
    "Belgaum": (15.8497, 74.4977),
    "Bellary": (15.1394, 76.9214),
    "Gulbarga": (17.3297, 76.8343),
    "Davangere": (14.4644, 75.9218),
    "Shimoga": (13.9310, 75.5681),
    "Tumkur": (13.3379, 77.1173),
    
    # Maharashtra
    "Nashik": (19.9975, 73.7898),
    "Aurangabad": (19.8762, 75.3433),
    "Solapur": (17.6599, 75.9064),
    "Kolhapur": (16.7050, 74.2433),
    "Sangli": (16.8544, 74.5815),
    "Satara": (17.6805, 74.0183),
    "Ahmednagar": (19.0948, 74.7480),
    "Nanded": (19.1383, 77.3210),
    "Amravati": (20.9320, 77.7523),
    
    # Rajasthan
    "Jodhpur": (26.2389, 73.0243),
    "Udaipur": (24.5854, 73.7125),
    "Kota": (25.2138, 75.8648),
    "Bikaner": (28.0229, 73.3119),
    "Ajmer": (26.4499, 74.6399),
    "Alwar": (27.5530, 76.6346),
    "Bhilwara": (25.3407, 74.6313),
    
    # Gujarat
    "Rajkot": (22.3039, 70.8022),
    "Bhavnagar": (21.7645, 72.1519),
    "Jamnagar": (22.4707, 70.0577),
    "Junagadh": (21.5222, 70.4579),
    "Gandhinagar": (23.2156, 72.6369),
    "Kutch": (23.7337, 69.8597),
    
    # Uttar Pradesh
    "Agra": (27.1767, 78.0081),
    "Varanasi": (25.3176, 82.9739),
    "Allahabad": (25.4358, 81.8463),
    "Prayagraj": (25.4358, 81.8463),
    "Meerut": (28.9845, 77.7064),
    "Ghaziabad": (28.6692, 77.4538),
    "Noida": (28.5355, 77.3910),
    "Gorakhpur": (26.7606, 83.3732),
    "Bareilly": (28.3670, 79.4304),
    "Aligarh": (27.8974, 78.0880),
    "Moradabad": (28.8389, 78.7768),
    
    # Bihar
    "Gaya": (24.7955, 85.0002),
    "Bhagalpur": (25.2425, 86.9842),
    "Muzaffarpur": (26.1209, 85.3647),
    "Darbhanga": (26.1542, 85.8918),
    "Purnia": (25.7771, 87.4753),
    
    # Madhya Pradesh
    "Gwalior": (26.2183, 78.1828),
    "Jabalpur": (23.1815, 79.9864),
    "Ujjain": (23.1765, 75.7885),
    "Sagar": (23.8388, 78.7378),
    "Rewa": (24.5362, 81.2989),
    "Satna": (24.5879, 80.8322),
    
    # West Bengal
    "Howrah": (22.5958, 88.2636),
    "Durgapur": (23.5204, 87.3119),
    "Siliguri": (26.7271, 88.6393),
    "Asansol": (23.6739, 86.9524),
    "Bardhaman": (23.2599, 87.8629),
    
    # Jharkhand
    "Ranchi": (23.3441, 85.3096),
    "Jamshedpur": (22.8046, 86.2029),
    "Dhanbad": (23.7957, 86.4304),
    "Bokaro": (23.6693, 86.1511),
    "Hazaribagh": (23.9966, 85.3591),
    
    # Odisha
    "Bhubaneswar": (20.2961, 85.8245),
    "Cuttack": (20.4625, 85.8830),
    "Rourkela": (22.2604, 84.8536),
    "Berhampur": (19.3149, 84.7941),
    "Sambalpur": (21.4669, 83.9756),
    
    # Chhattisgarh
    "Raipur": (21.2514, 81.6296),
    "Bilaspur": (22.0797, 82.1391),
    "Durg": (21.1904, 81.2849),
    "Korba": (22.3595, 82.7501),
    
    # Punjab
    "Amritsar": (31.6340, 74.8723),
    "Ludhiana": (30.9010, 75.8573),
    "Jalandhar": (31.3260, 75.5762),
    "Patiala": (30.3398, 76.3869),
    "Bathinda": (30.2110, 74.9455),
    
    # Haryana  
    "Faridabad": (28.4089, 77.3178),
    "Gurugram": (28.4595, 77.0266),
    "Gurgaon": (28.4595, 77.0266),
    "Ambala": (30.3782, 76.7767),
    "Panipat": (29.3909, 76.9635),
    "Karnal": (29.6857, 76.9905),
    "Rohtak": (28.8955, 76.6066),
    "Hisar": (29.1492, 75.7217),
    
    # Assam
    "Guwahati": (26.1445, 91.7362),
    "Kamrup": (26.1445, 91.7362),
    "Dibrugarh": (27.4728, 94.9120),
    "Jorhat": (26.7509, 94.2037),
    "Silchar": (24.8333, 92.7789),
    "Nagaon": (26.3509, 92.6837),
    
    # Northeast
    "Shillong": (25.5788, 91.8933),
    "Imphal": (24.8170, 93.9368),
    "Agartala": (23.8315, 91.2868),
    "Aizawl": (23.7271, 92.7176),
    "Kohima": (25.6701, 94.1077),
    "Itanagar": (27.0844, 93.6053),
    "Gangtok": (27.3389, 88.6065),
    
    # Uttarakhand
    "Dehradun": (30.3165, 78.0322),
    "Haridwar": (29.9457, 78.1642),
    "Nainital": (29.3919, 79.4542),
    "Udham Singh Nagar": (28.9800, 79.4100),
    
    # Himachal Pradesh
    "Shimla": (31.1048, 77.1734),
    "Kangra": (32.0998, 76.2691),
    "Mandi": (31.7088, 76.9320),
    "Kullu": (31.9592, 77.1089),
    "Solan": (30.9045, 77.0967),
    
    # Goa
    "North Goa": (15.5479, 73.8013),
    "South Goa": (15.1839, 74.0510),
    
    # Chandigarh
    "Chandigarh": (30.7333, 76.7794),
    
    # Telangana
    "Warangal": (17.9784, 79.5941),
    "Karimnagar": (18.4386, 79.1288),
    "Nizamabad": (18.6725, 78.0941),
    "Khammam": (17.2473, 80.1514),
    "Nalgonda": (17.0575, 79.2690),
    "Rangareddy": (17.2543, 78.1298),
    "Medak": (18.0526, 78.2628),
    "Adilabad": (19.6641, 78.5320),
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
    "Chandigarh": (30.7333, 76.7794),
    "Jammu And Kashmir": (33.7782, 76.5762),
    "Ladakh": (34.1526, 77.5771),
    "Puducherry": (11.9416, 79.8083),
    "A & N Islands": (11.7401, 92.6586),
    "D & N Haveli And Daman & Diu": (20.1809, 73.0169),
    "Lakshadweep": (10.5667, 72.6417),
}

def get_coordinates(district, state):
    """Get coordinates for a district with fallbacks"""
    import random
    
    # Try exact match
    if district in DISTRICT_COORDS:
        return DISTRICT_COORDS[district]
    
    # Try fuzzy match
    district_lower = district.lower()
    for key, coords in DISTRICT_COORDS.items():
        if key.lower() == district_lower:
            return coords
        if key.lower() in district_lower or district_lower in key.lower():
            return coords
    
    # Use state center with random offset
    if state in STATE_CENTERS:
        lat, lng = STATE_CENTERS[state]
        offset_lat = random.uniform(-1.0, 1.0)
        offset_lng = random.uniform(-1.0, 1.0)
        return (lat + offset_lat, lng + offset_lng)
    
    return None

def main():
    print("=" * 60)
    print("🚀 Seeding Database with XGBoost Predictions + Kerala 2024")
    print("=" * 60)
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ ERROR: DATABASE_URL not found in .env")
        return
    
    # Name mappings for display (NCRB name → friendly name)
    NAME_MAPPINGS = {
        "Ernakulam City": "Kochi",
        "Ernakulam Commr.": "Kochi",
        "Thiruvananthapuram City": "Trivandrum",
        "Kozhikode City": "Calicut",
    }
    
    # Kerala 2024 data path
    KERALA_2024_PATH = os.path.join(BASE_DIR, "..", "frontend", "crime_data", "kerala_district_crime_2024.csv")
    
    # Read XGBoost predictions
    print("\n📖 Reading XGBoost predictions...")
    districts = []
    existing_names = set()
    
    with open(PREDICTIONS_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            district = row['district']
            state = row['state']
            
            coords = get_coordinates(district, state)
            if not coords:
                continue
            
            # Apply name mapping for display
            display_name = NAME_MAPPINGS.get(district, district)
            
            districts.append({
                'name': display_name,
                'state': state,
                'latitude': coords[0],
                'longitude': coords[1],
                'safety_zone': row['xgboost_zone'],
                'crime_index': float(row['crime_score']),
                'confidence': float(row['confidence'])
            })
            existing_names.add(display_name.lower())
    
    print(f"✅ Loaded {len(districts)} districts from XGBoost predictions")
    
    # Add Kerala 2024 data for districts not already in the list
    kerala_added = 0
    if os.path.exists(KERALA_2024_PATH):
        print("\n📖 Adding Kerala 2024 data...")
        with open(KERALA_2024_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                district = row['District'].strip()
                
                # Apply name mapping
                display_name = NAME_MAPPINGS.get(district, district)
                
                # Skip if already exists (case-insensitive check)
                if display_name.lower() in existing_names:
                    continue
                
                coords = get_coordinates(district, "Kerala")
                if not coords:
                    continue
                
                # Map risk labels to zones
                risk_label = row.get('Risk_Label_Inferred', 'Moderate Risk')
                if 'High' in risk_label:
                    zone = 'red'
                elif 'Safe' in risk_label:
                    zone = 'green'
                else:
                    zone = 'orange'
                
                # Calculate crime_index from FIR count
                try:
                    fir_count = int(row.get('Total_FIRs', 0))
                    crime_index = min(100, fir_count / 500)  # Normalize
                except:
                    crime_index = 50.0
                
                districts.append({
                    'name': display_name,
                    'state': 'Kerala',
                    'latitude': coords[0],
                    'longitude': coords[1],
                    'safety_zone': zone,
                    'crime_index': crime_index,
                    'confidence': 0.85  # Kerala 2024 data confidence
                })
                existing_names.add(display_name.lower())
                kerala_added += 1
        
        print(f"   ➕ Added {kerala_added} Kerala districts from 2024 data")
    
    print(f"\n✅ Total: {len(districts)} districts with coordinates")
    
    # Count by zone
    zone_counts = {'green': 0, 'orange': 0, 'red': 0}
    for d in districts:
        zone_counts[d['safety_zone']] += 1
    
    print(f"\n📊 Distribution:")
    print(f"   🟢 Safe: {zone_counts['green']}")
    print(f"   🟠 Moderate: {zone_counts['orange']}")
    print(f"   🔴 High Risk: {zone_counts['red']}")
    
    # Connect to database
    print("\n🔄 Connecting to database...")
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()
    
    # Clear existing data
    cur.execute("DELETE FROM cities")
    print("🗑️ Cleared existing city data")
    
    # Check if crime_index column exists, add if not
    try:
        cur.execute("ALTER TABLE cities ADD COLUMN IF NOT EXISTS crime_index FLOAT")
        cur.execute("ALTER TABLE cities ADD COLUMN IF NOT EXISTS confidence FLOAT")
        conn.commit()
    except:
        pass
    
    # Insert new data
    inserted = 0
    for d in districts:
        try:
            cur.execute("""
                INSERT INTO cities (name, state, latitude, longitude, safety_zone, crime_index)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (d['name'], d['state'], d['latitude'], d['longitude'], 
                  d['safety_zone'], d['crime_index']))
            inserted += 1
        except Exception as e:
            print(f"⚠️ Error inserting {d['name']}: {e}")
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"\n✅ Successfully inserted {inserted} XGBoost-classified districts!")
    
    # Print sample
    print("\n📋 Sample entries:")
    for d in districts[:5]:
        print(f"   {d['name']}, {d['state']} - {d['safety_zone'].upper()} (confidence: {d['confidence']:.1%})")

if __name__ == "__main__":
    main()
