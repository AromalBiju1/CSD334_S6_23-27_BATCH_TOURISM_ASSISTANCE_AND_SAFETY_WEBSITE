"""
Comprehensive Safety Data Seed Script
=====================================
Seeds cities database with:
1. NCRB district-level crime data (972 districts)
2. Conflict zone overlays (J&K, Naxal belt, Northeast)
3. Recent incident overrides (Pahalgam, etc.)
4. Proper risk classification based on actual crime rates

Run from backend directory with venv activated:
    python seed_comprehensive.py
"""

import sys
import os
from pathlib import Path
import csv
import json

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
CRIME_DATA_DIR = SCRIPT_DIR.parent / "frontend" / "crime_data"
NCRB_DATA = CRIME_DATA_DIR / "NCRB_District_Table_1.1.csv"
DISTRICT_CRIME = CRIME_DATA_DIR / "india_district_crime_2022.csv"
TOURIST_DATA = CRIME_DATA_DIR / "india_tourist_destinations_full.csv"

# Import from database package
from database.database import sessionLocal, engine, Base
from database.models import City, EmergencyContact

# ============== CONFLICT ZONES ==============
# Areas with known terrorism/insurgency concerns - force HIGH RISK
CONFLICT_ZONES = {
    # Jammu & Kashmir - All districts marked sensitive
    "Jammu_Kashmir": [
        "Anantnag", "Pulwama", "Shopian", "Kulgam", "Budgam", "Srinagar",
        "Ganderbal", "Baramulla", "Kupwara", "Bandipora", "Pahalgam",
        "Rajouri", "Poonch", "Kishtwar", "Ramban", "Doda"
    ],
    # Naxal Belt - Left-wing extremism
    "Naxal_Belt": [
        "Bastar", "Dantewada", "Sukma", "Bijapur", "Narayanpur", "Kondagaon",
        "Kanker", "Gadchiroli", "Gondia", "Latehar", "Garhwa", "Palamu",
        "Chatra", "Bokaro", "Giridih", "Gumla", "Lohardaga", "Khunti",
        "West_Singhbhum", "Koraput", "Malkangiri", "Rayagada"
    ],
    # Northeast Insurgency
    "Northeast": [
        "Imphal_East", "Imphal_West", "Thoubal", "Bishnupur", "Churachandpur",
        "Chandel", "Ukhrul", "Senapati", "Tamenglong", "Noney",
        "Mon", "Tuensang", "Longleng", "Kiphire", "Zunheboto",
        "Tinsukia", "Dibrugarh", "Karbi_Anglong"
    ],
    # Border Sensitivity
    "Border_Areas": [
        "Pathankot", "Gurdaspur", "Amritsar", "Ferozepur",
        "Kutch", "Jaisalmer", "Ganganagar", "Barmer"
    ]
}

# Recent terrorist incidents - override to HIGH RISK
RECENT_INCIDENTS = {
    "Pahalgam": {"zone": "red", "reason": "2025 terrorist attack", "crime_index": 100},
    "Pulwama": {"zone": "red", "reason": "Ongoing security concerns", "crime_index": 95},
    "Shopian": {"zone": "red", "reason": "Security sensitive", "crime_index": 90},
    "Kupwara": {"zone": "red", "reason": "Border area", "crime_index": 85},
}

# ============== DISTRICT COORDINATES ==============
# Approximate coordinates for major districts (lat, lng)
# In production, use a geocoding API or official Census data
DISTRICT_COORDS = {
    # Andhra Pradesh
    "Anantapur": (14.6819, 77.6006),
    "Chittoor": (13.2172, 79.1003),
    "East_Godavari": (17.3212, 82.0438),
    "Guntur": (16.3067, 80.4365),
    "Krishna": (16.6100, 80.7214),
    "Kurnool": (15.8281, 78.0373),
    "Nellore": (14.4426, 79.9865),
    "Prakasam": (15.3467, 79.5542),
    "Srikakulam": (18.2949, 83.8938),
    "Visakhapatnam": (17.6868, 83.2185),
    "Vizianagaram": (18.1067, 83.3956),
    "West_Godavari": (16.9174, 81.3399),
    "YSR_Kadapa": (14.4673, 78.8242),
    
    # Arunachal Pradesh
    "Itanagar": (27.0844, 93.6053),
    "Papum_Pare": (27.0511, 93.5344),
    "West_Kameng": (27.5000, 92.0000),
    
    # Assam
    "Dibrugarh": (27.4728, 94.9120),
    "Guwahati_City": (26.1445, 91.7362),
    "Jorhat": (26.7509, 94.2037),
    "Kamrup": (26.0000, 91.5000),
    "Nagaon": (26.3465, 92.6855),
    "Sivasagar": (26.9826, 94.6381),
    "Tinsukia": (27.4887, 95.3558),
    
    # Bihar
    "Patna": (25.5941, 85.1376),
    "Gaya": (24.7964, 85.0062),
    "Muzaffarpur": (26.1209, 85.3647),
    "Bhagalpur": (25.2425, 86.9842),
    "Darbhanga": (26.1542, 85.8918),
    "Aurangabad": (24.7520, 84.3739),
    "Begusarai": (25.4182, 86.1272),
    
    # Chhattisgarh
    "Raipur": (21.2514, 81.6296),
    "Bilaspur": (22.0797, 82.1391),
    "Durg": (21.1904, 81.2849),
    "Korba": (22.3595, 82.7501),
    "Raigarh": (21.8974, 83.3950),
    "Rajnandgaon": (21.0974, 81.0285),
    "Bastar": (19.0748, 81.9540),
    "Dantewada": (18.8974, 81.3480),
    "Sukma": (18.3974, 81.6480),
    
    # Delhi
    "Central": (28.6490, 77.2112),
    "East": (28.6279, 77.2951),
    "New_Delhi": (28.6139, 77.2090),
    "North": (28.7041, 77.1025),
    "North_East": (28.6947, 77.2951),
    "North_West": (28.7167, 77.0833),
    "South": (28.5244, 77.2167),
    "South_East": (28.5503, 77.3061),
    "South_West": (28.5800, 77.0300),
    "West": (28.6333, 77.0167),
    
    # Goa
    "North_Goa": (15.5301, 73.8151),
    "South_Goa": (15.2004, 74.1234),
    
    # Gujarat
    "Ahmedabad": (23.0225, 72.5714),
    "Surat": (21.1702, 72.8311),
    "Vadodara": (22.3072, 73.1812),
    "Rajkot": (22.3039, 70.8022),
    "Bhavnagar": (21.7645, 72.1519),
    "Jamnagar": (22.4707, 70.0577),
    "Junagadh": (21.5222, 70.4579),
    "Gandhinagar": (23.2156, 72.6369),
    "Kutch": (23.7337, 69.8597),
    
    # Haryana
    "Gurugram": (28.4595, 77.0266),
    "Faridabad": (28.4089, 77.3178),
    "Panipat": (29.3909, 76.9635),
    "Ambala": (30.3782, 76.7767),
    "Karnal": (29.6857, 76.9905),
    "Sonipat": (28.9288, 77.0913),
    "Rohtak": (28.8955, 76.6066),
    "Hissar": (29.1492, 75.7217),
    
    # Himachal Pradesh
    "Shimla": (31.1048, 77.1734),
    "Kullu": (31.9579, 77.1095),
    "Manali": (32.2432, 77.1892),
    "Dharamshala": (32.2190, 76.3234),
    "Kangra": (32.0998, 76.2691),
    "Mandi": (31.7082, 76.9316),
    
    # Jammu & Kashmir (HIGH RISK ZONE)
    "Srinagar": (34.0837, 74.7973),
    "Jammu": (32.7266, 74.8570),
    "Anantnag": (33.7311, 75.1593),
    "Baramulla": (34.2023, 74.3563),
    "Pahalgam": (34.0161, 75.3150),
    "Gulmarg": (34.0484, 74.3805),
    "Pulwama": (33.8748, 74.8952),
    "Shopian": (33.7162, 74.8288),
    "Kupwara": (34.5263, 74.2543),
    "Kulgam": (33.6450, 75.0194),
    "Budgam": (33.8978, 74.7157),
    "Ganderbal": (34.2261, 74.7790),
    "Bandipora": (34.4181, 74.6406),
    "Rajouri": (33.3792, 74.3110),
    "Poonch": (33.7715, 74.0960),
    "Kishtwar": (33.3112, 75.7662),
    "Doda": (33.1492, 75.5475),
    "Ramban": (33.2456, 75.2365),
    
    # Ladakh (Border area)
    "Leh": (34.1526, 77.5771),
    "Kargil": (34.5539, 76.1349),
    
    # Kerala
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Kochi": (9.9312, 76.2673),
    "Kozhikode": (11.2588, 75.7804),
    "Thrissur": (10.5276, 76.2144),
    "Kollam": (8.8932, 76.6141),
    "Kannur": (11.8745, 75.3704),
    "Alappuzha": (9.4981, 76.3388),
    "Palakkad": (10.7867, 76.6548),
    "Malappuram": (11.0509, 76.0711),
    "Idukki": (9.8494, 76.9710),
    "Munnar": (10.0889, 77.0595),
    "Fort_Kochi": (9.9658, 76.2421),
    
    # Madhya Pradesh
    "Bhopal": (23.2599, 77.4126),
    "Indore": (22.7196, 75.8577),
    "Jabalpur": (23.1815, 79.9864),
    "Gwalior": (26.2183, 78.1828),
    "Ujjain": (23.1765, 75.7885),
    "Sagar": (23.8388, 78.7378),
    "Rewa": (24.5362, 81.3037),
    "Satna": (24.6005, 80.8322),
    
    # Maharashtra
    "Mumbai": (19.0760, 72.8777),
    "Pune": (18.5204, 73.8567),
    "Nagpur": (21.1458, 79.0882),
    "Nashik": (19.9975, 73.7898),
    "Aurangabad": (19.8762, 75.3433),
    "Thane": (19.2183, 72.9781),
    "Solapur": (17.6599, 75.9064),
    "Kolhapur": (16.7050, 74.2433),
    "Amravati": (20.9320, 77.7523),
    
    # Rajasthan
    "Jaipur": (26.9124, 75.7873),
    "Jodhpur": (26.2389, 73.0243),
    "Udaipur": (24.5854, 73.7125),
    "Kota": (25.2138, 75.8648),
    "Ajmer": (26.4499, 74.6399),
    "Bikaner": (28.0229, 73.3119),
    "Jaisalmer": (26.9157, 70.9083),
    "Pushkar": (26.4897, 74.5511),
    "Mount_Abu": (24.5926, 72.7156),
    
    # Tamil Nadu
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    "Tiruchirappalli": (10.7905, 78.7047),
    "Salem": (11.6643, 78.1460),
    "Tirunelveli": (8.7139, 77.7567),
    "Vellore": (12.9165, 79.1325),
    "Thanjavur": (10.7870, 79.1378),
    "Ooty": (11.4102, 76.6950),
    "Kanyakumari": (8.0883, 77.5385),
    "Mahabalipuram": (12.6269, 80.1929),
    "Pondicherry": (11.9416, 79.8083),
    
    # Uttar Pradesh
    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
    "Agra": (27.1767, 78.0081),
    "Varanasi": (25.3176, 82.9739),
    "Allahabad": (25.4358, 81.8463),
    "Gorakhpur": (26.7606, 83.3732),
    "Mathura": (27.4924, 77.6737),
    "Noida": (28.5355, 77.3910),
    "Ghaziabad": (28.6692, 77.4538),
    "Ayodhya": (26.7922, 82.1998),
    
    # Uttarakhand
    "Dehradun": (30.3165, 78.0322),
    "Haridwar": (29.9457, 78.1642),
    "Rishikesh": (30.0869, 78.2676),
    "Nainital": (29.3919, 79.4542),
    "Mussoorie": (30.4598, 78.0644),
    
    # West Bengal
    "Kolkata": (22.5726, 88.3639),
    "Darjeeling": (27.0410, 88.2663),
    "Siliguri": (26.7271, 88.6393),
    "Howrah": (22.5958, 88.2636),
    "Asansol": (23.6739, 86.9523),
}

# State centroids for districts without specific coordinates
STATE_CENTROIDS = {
    "Andhra_Pradesh": (15.9129, 79.7400),
    "Arunachal_Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chhattisgarh": (21.2787, 81.8661),
    "Delhi": (28.6139, 77.2090),
    "Goa": (15.2993, 74.1240),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal_Pradesh": (31.1048, 77.1734),
    "Jammu_Kashmir": (33.7782, 76.5762),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya_Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Sikkim": (27.5330, 88.5122),
    "Tamil_Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Tripura": (23.9408, 91.9882),
    "Uttar_Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West_Bengal": (22.9868, 87.8550),
    "Ladakh": (34.2268, 77.5619),
}


def get_district_coords(district_name: str, state_name: str) -> tuple:
    """Get coordinates for a district, falling back to state centroid."""
    # Clean names
    clean_district = district_name.replace(" ", "_").replace("-", "_")
    clean_state = state_name.replace(" ", "_").replace("-", "_")
    
    # Try district coordinates
    if clean_district in DISTRICT_COORDS:
        return DISTRICT_COORDS[clean_district]
    
    # Try alternate forms
    for key, coords in DISTRICT_COORDS.items():
        if clean_district.lower() in key.lower() or key.lower() in clean_district.lower():
            return coords
    
    # Fall back to state centroid with small offset
    if clean_state in STATE_CENTROIDS:
        lat, lng = STATE_CENTROIDS[clean_state]
        # Add small random offset to avoid exact overlaps
        import random
        offset = random.uniform(-0.3, 0.3)
        return (lat + offset, lng + offset)
    
    # Default to center of India
    return (20.5937, 78.9629)


def is_conflict_zone(district_name: str) -> bool:
    """Check if district is in a conflict zone."""
    clean_name = district_name.replace(" ", "_").replace("-", "_")
    for zone, districts in CONFLICT_ZONES.items():
        for d in districts:
            if clean_name.lower() == d.lower() or d.lower() in clean_name.lower():
                return True
    return False


def get_incident_override(district_name: str) -> dict | None:
    """Check if district has a recent incident override."""
    clean_name = district_name.replace(" ", "_").replace("-", "_")
    for name, data in RECENT_INCIDENTS.items():
        if clean_name.lower() == name.lower() or name.lower() in clean_name.lower():
            return data
    return None


def calculate_safety_zone(crime_rate: float, is_conflict: bool, incident: dict | None) -> tuple:
    """
    Calculate safety zone and crime index.
    Returns: (safety_zone, crime_index)
    """
    # Override for recent incidents
    if incident:
        return (incident["zone"], incident["crime_index"])
    
    # Conflict zones capped at orange (moderate) minimum
    if is_conflict:
        return ("red", 85.0)  # Force high risk for conflict zones
    
    # Crime rate based classification
    # Using per lakh population rates from NCRB
    if crime_rate <= 150:
        return ("green", crime_rate / 5)  # Safe
    elif crime_rate <= 300:
        return ("orange", 30 + (crime_rate - 150) / 5)  # Moderate
    else:
        return ("red", 60 + min((crime_rate - 300) / 10, 40))  # High


def seed_from_district_crime():
    """Seed from india_district_crime_2022.csv which has 198 districts with risk labels."""
    db = sessionLocal()
    cities_added = 0
    seen = set()
    
    try:
        if not DISTRICT_CRIME.exists():
            print(f"File not found: {DISTRICT_CRIME}")
            return 0
        
        print(f"Loading from {DISTRICT_CRIME}")
        with open(DISTRICT_CRIME, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                state = row.get('State', '').strip()
                district = row.get('District', '').strip()
                
                if not district or not state or district == 'Total_Districts':
                    continue
                
                key = f"{district}_{state}".lower()
                if key in seen:
                    continue
                seen.add(key)
                
                # Get coordinates
                lat, lng = get_district_coords(district, state)
                
                # Get crime data
                try:
                    crime_rate = float(row.get('Crime_Rate', 200))
                except (ValueError, TypeError):
                    crime_rate = 200
                
                # Check for conflict zone and incidents
                is_conflict = is_conflict_zone(district)
                incident = get_incident_override(district)
                
                # Calculate safety
                risk_label = row.get('Risk_Label', 'Moderate')
                
                # Override based on conflict/incidents
                if incident:
                    safety_zone = incident["zone"]
                    crime_index = incident["crime_index"]
                elif is_conflict:
                    safety_zone = "red"
                    crime_index = 85.0
                else:
                    # Use the risk label from CSV
                    zone_map = {"Safe": "green", "Moderate": "orange", "High": "red"}
                    safety_zone = zone_map.get(risk_label, "orange")
                    crime_index = crime_rate / 4  # Normalize
                
                city = City(
                    name=district.replace("_", " "),
                    state=state.replace("_", " "),
                    latitude=lat,
                    longitude=lng,
                    population=None,
                    crime_index=min(crime_index, 100),
                    safety_zone=safety_zone
                )
                db.add(city)
                cities_added += 1
        
        db.commit()
        print(f"Added {cities_added} districts from crime data")
        return cities_added, seen
        
    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 0, set()
    finally:
        db.close()


def seed_tourist_destinations(seen: set):
    """Add tourist destinations not already in district list."""
    db = sessionLocal()
    added = 0
    
    try:
        if not TOURIST_DATA.exists():
            print(f"File not found: {TOURIST_DATA}")
            return 0
        
        print(f"Loading tourist destinations from {TOURIST_DATA}")
        with open(TOURIST_DATA, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get('Location', '').replace('_', ' ')
                state = row.get('State', '').replace('_', ' ')
                
                if not name:
                    continue
                
                key = f"{name}_{state}".lower().replace(" ", "_")
                if key in seen:
                    continue
                seen.add(key)
                
                try:
                    lat = float(row.get('Latitude', 0))
                    lng = float(row.get('Longitude', 0))
                except (ValueError, TypeError):
                    continue
                
                if lat == 0 or lng == 0:
                    continue
                
                # Check for conflict zone
                is_conflict = is_conflict_zone(name)
                incident = get_incident_override(name)
                
                risk_label = row.get('Risk_Label', 'Moderate')
                
                if incident:
                    safety_zone = incident["zone"]
                    crime_index = incident["crime_index"]
                elif is_conflict:
                    safety_zone = "red"
                    crime_index = 85.0
                else:
                    zone_map = {"Safe": "green", "Moderate": "orange", "High Risk": "red"}
                    safety_zone = zone_map.get(risk_label, "orange")
                    crime_index = {"green": 25, "orange": 50, "red": 80}.get(safety_zone, 50)
                
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
                added += 1
        
        db.commit()
        print(f"Added {added} tourist destinations")
        return added
        
    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        return 0
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
            ("Anti-Terror (NSG)", "1600-100-700", "security", True),
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
    finally:
        db.close()


def seed_city_aliases():
    """Add popular city name aliases that people commonly search for."""
    db = sessionLocal()
    added = 0
    
    # Common name -> (Official district name, state, lat, lng, safety_zone, crime_index)
    CITY_ALIASES = [
        # Kerala
        ("Kochi", "Kerala", 9.9312, 76.2673, "orange", 50.0),
        ("Cochin", "Kerala", 9.9312, 76.2673, "orange", 50.0),
        ("Trivandrum", "Kerala", 8.5241, 76.9366, "red", 75.0),
        ("Calicut", "Kerala", 11.2588, 75.7804, "red", 75.0),
        ("Trichur", "Kerala", 10.5276, 76.2144, "red", 70.0),
        ("Wayanad Hill Station", "Kerala", 11.6854, 76.1320, "orange", 50.0),
        
        # Major metros - common names
        ("Bengaluru", "Karnataka", 12.9716, 77.5946, "red", 80.0),
        ("Bangalore", "Karnataka", 12.9716, 77.5946, "red", 80.0),
        ("Bombay", "Maharashtra", 19.0760, 72.8777, "red", 80.0),
        ("Calcutta", "West Bengal", 22.5726, 88.3639, "red", 75.0),
        ("Madras", "Tamil Nadu", 13.0827, 80.2707, "red", 80.0),
        
        # Popular tourist cities not in district data
        ("Pondicherry", "Puducherry", 11.9416, 79.8083, "green", 25.0),
        ("Puducherry", "Puducherry", 11.9416, 79.8083, "green", 25.0),
        ("Agra City", "Uttar Pradesh", 27.1767, 78.0081, "orange", 60.0),
        ("Jaipur Pink City", "Rajasthan", 26.9124, 75.7873, "orange", 60.0),
        ("Vrindavan", "Uttar Pradesh", 27.5803, 77.6966, "green", 30.0),
        ("Mathura City", "Uttar Pradesh", 27.4924, 77.6737, "orange", 50.0),
        
        # Hill stations
        ("Nainital Lake", "Uttarakhand", 29.3919, 79.4542, "green", 25.0),
        ("Mussoorie Hills", "Uttarakhand", 30.4598, 78.0644, "green", 25.0),
        ("Shimla Hills", "Himachal Pradesh", 31.1048, 77.1734, "orange", 50.0),
        ("Manali Valley", "Himachal Pradesh", 32.2432, 77.1892, "green", 25.0),
        ("Ooty Hills", "Tamil Nadu", 11.4102, 76.6950, "green", 25.0),
        ("Kodaikanal Hills", "Tamil Nadu", 10.2381, 77.4892, "green", 25.0),
        
        # Beaches
        ("Goa Beaches", "Goa", 15.2993, 74.1240, "orange", 50.0),
        ("Puri Beach", "Odisha", 19.7983, 85.8249, "orange", 50.0),
        ("Diu Island", "Daman and Diu", 20.7141, 70.9874, "green", 25.0),
        
        # Spiritual
        ("Haridwar Ghats", "Uttarakhand", 29.9457, 78.1642, "orange", 50.0),
        ("Rishikesh Yoga", "Uttarakhand", 30.0869, 78.2676, "green", 25.0),
        ("Tirupati", "Andhra Pradesh", 13.6288, 79.4192, "orange", 50.0),
        ("Shirdi Temple", "Maharashtra", 19.7645, 74.4773, "green", 25.0),
        ("Amritsar Golden Temple", "Punjab", 31.6200, 74.8765, "orange", 60.0),
    ]
    
    try:
        for name, state, lat, lng, zone, crime_idx in CITY_ALIASES:
            # Check if already exists
            existing = db.query(City).filter(City.name == name, City.state == state).first()
            if not existing:
                city = City(
                    name=name,
                    state=state,
                    latitude=lat,
                    longitude=lng,
                    population=None,
                    crime_index=crime_idx,
                    safety_zone=zone
                )
                db.add(city)
                added += 1
        
        db.commit()
        print(f"Added {added} city aliases")
        return added
        
    except Exception as e:
        db.rollback()
        print(f"Error adding aliases: {e}")
        return 0
    finally:
        db.close()


def print_stats():
    """Print database statistics."""
    db = sessionLocal()
    try:
        total = db.query(City).count()
        green = db.query(City).filter(City.safety_zone == "green").count()
        orange = db.query(City).filter(City.safety_zone == "orange").count()
        red = db.query(City).filter(City.safety_zone == "red").count()
        
        print("\n" + "=" * 50)
        print("DATABASE STATISTICS")
        print("=" * 50)
        print(f"Total locations: {total}")
        print(f"  🟢 Safe (green):     {green} ({100*green/total:.1f}%)")
        print(f"  🟠 Moderate (orange): {orange} ({100*orange/total:.1f}%)")
        print(f"  🔴 High Risk (red):   {red} ({100*red/total:.1f}%)")
        print("=" * 50)
        
        # Show some conflict zone examples
        print("\nConflict Zone Examples:")
        conflicts = db.query(City).filter(City.safety_zone == "red").limit(10).all()
        for c in conflicts:
            print(f"  🔴 {c.name}, {c.state} - Crime Index: {c.crime_index}")
        
    finally:
        db.close()


def main():
    print("=" * 60)
    print("COMPREHENSIVE SAFETY DATA SEEDER")
    print("=" * 60)
    print(f"Crime data directory: {CRIME_DATA_DIR}")
    print()
    
    # Clear existing data
    db = sessionLocal()
    db.query(City).delete()
    db.commit()
    db.close()
    print("Cleared existing cities")
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Seed districts first
    district_count, seen = seed_from_district_crime()
    
    # Add tourist destinations
    tourist_count = seed_tourist_destinations(seen)
    
    # Add emergency contacts
    seed_emergency_contacts()
    
    # Print stats
    print_stats()
    
    total = district_count + tourist_count
    print(f"\n✅ COMPLETE: {total} locations seeded")
    print("   A* algorithm will now AVOID red zones completely!")


if __name__ == "__main__":
    main()
