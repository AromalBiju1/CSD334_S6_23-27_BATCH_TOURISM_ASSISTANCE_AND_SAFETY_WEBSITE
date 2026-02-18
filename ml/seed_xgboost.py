"""
XGBoost-Based District Seeder
==============================
Uses the trained XGBoost classifier to classify all districts
and seed them into the database with proper safety zones.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json

# Add backend to path
SCRIPT_DIR = Path(__file__).parent.parent
BACKEND_DIR = SCRIPT_DIR.parent / "backend"
ML_DIR = SCRIPT_DIR / "ml"
sys.path.insert(0, str(BACKEND_DIR))

# Import after path setup
from database.database import sessionLocal, engine
from database.models import Base, City, EmergencyContact

# District coordinates (lat, lng) - approximations for most districts
# We'll use state-level averages where specific coordinates aren't known
STATE_COORDS = {
    'Andhra Pradesh': (15.9129, 79.7400),
    'Arunachal Pradesh': (28.2180, 94.7278),
    'Assam': (26.2006, 92.9376),
    'Bihar': (25.0961, 85.3131),
    'Chhattisgarh': (21.2787, 81.8661),
    'Goa': (15.2993, 74.1240),
    'Gujarat': (22.2587, 71.1924),
    'Haryana': (29.0588, 76.0856),
    'Himachal Pradesh': (31.1048, 77.1734),
    'Jharkhand': (23.6102, 85.2799),
    'Karnataka': (15.3173, 75.7139),
    'Kerala': (10.8505, 76.2711),
    'Madhya Pradesh': (22.9734, 78.6569),
    'Maharashtra': (19.7515, 75.7139),
    'Manipur': (24.6637, 93.9063),
    'Meghalaya': (25.4670, 91.3662),
    'Mizoram': (23.1645, 92.9376),
    'Nagaland': (26.1584, 94.5624),
    'Odisha': (20.9517, 85.0985),
    'Punjab': (31.1471, 75.3412),
    'Rajasthan': (27.0238, 74.2179),
    'Sikkim': (27.5330, 88.5122),
    'Tamil Nadu': (11.1271, 78.6569),
    'Telangana': (17.8686, 79.0141),
    'Tripura': (23.9408, 91.9882),
    'Uttar Pradesh': (26.8467, 80.9462),
    'Uttarakhand': (30.0668, 79.0193),
    'West Bengal': (22.9868, 87.8550),
    'Andaman and Nicobar Islands': (11.7401, 92.6586),
    'Chandigarh': (30.7333, 76.7794),
    'Dadra and Nagar Haveli and Daman and Diu': (20.3974, 72.8328),
    'Delhi': (28.7041, 77.1025),
    'Jammu and Kashmir': (33.2778, 75.3412),
    'Ladakh': (34.1526, 77.5771),
    'Lakshadweep': (10.5667, 72.6417),
    'Puducherry': (11.9416, 79.8083),
    'A & N Islands': (11.7401, 92.6586),
    'D & N Haveli': (20.2667, 73.0167),
    'Daman & Diu': (20.4283, 72.8397),
}

# Specific district coordinates (major districts)
DISTRICT_COORDS = {
    # Kerala
    ('Kerala', 'Thiruvananthapuram'): (8.5241, 76.9366),
    ('Kerala', 'Ernakulam'): (9.9312, 76.2673),
    ('Kerala', 'Kozhikode'): (11.2588, 75.7804),
    ('Kerala', 'Thrissur'): (10.5276, 76.2144),
    ('Kerala', 'Kollam'): (8.8932, 76.6141),
    ('Kerala', 'Kannur'): (11.8745, 75.3704),
    ('Kerala', 'Alappuzha'): (9.4981, 76.3388),
    ('Kerala', 'Palakkad'): (10.7867, 76.6548),
    ('Kerala', 'Malappuram'): (11.0509, 76.0711),
    ('Kerala', 'Idukki'): (9.8494, 76.9710),
    ('Kerala', 'Wayanad'): (11.6854, 76.1320),
    ('Kerala', 'Kottayam'): (9.5916, 76.5222),
    ('Kerala', 'Kasaragod'): (12.4996, 74.9869),
    ('Kerala', 'Pathanamthitta'): (9.2648, 76.7870),
    
    # Major metros and cities
    ('Maharashtra', 'Mumbai'): (19.0760, 72.8777),
    ('Maharashtra', 'Mumbai City'): (18.9388, 72.8354),
    ('Maharashtra', 'Mumbai Suburban'): (19.1234, 72.8361),
    ('Maharashtra', 'Pune'): (18.5204, 73.8567),
    ('Maharashtra', 'Nagpur'): (21.1458, 79.0882),
    ('Karnataka', 'Bengaluru Urban'): (12.9716, 77.5946),
    ('Karnataka', 'Bengaluru Rural'): (13.0530, 77.3875),
    ('Karnataka', 'Mysuru'): (12.2958, 76.6394),
    ('Tamil Nadu', 'Chennai'): (13.0827, 80.2707),
    ('Tamil Nadu', 'Coimbatore'): (11.0168, 76.9558),
    ('Tamil Nadu', 'Madurai'): (9.9252, 78.1198),
    ('West Bengal', 'Kolkata'): (22.5726, 88.3639),
    ('Delhi', 'Delhi'): (28.7041, 77.1025),
    ('Delhi', 'New Delhi'): (28.6139, 77.2090),
    ('Telangana', 'Hyderabad'): (17.3850, 78.4867),
    ('Gujarat', 'Ahmedabad'): (23.0225, 72.5714),
    ('Gujarat', 'Surat'): (21.1702, 72.8311),
    ('Rajasthan', 'Jaipur'): (26.9124, 75.7873),
    ('Uttar Pradesh', 'Lucknow'): (26.8467, 80.9462),
    ('Uttar Pradesh', 'Varanasi'): (25.3176, 82.9739),
    ('Uttar Pradesh', 'Agra'): (27.1767, 78.0081),
    ('Madhya Pradesh', 'Bhopal'): (23.2599, 77.4126),
    ('Madhya Pradesh', 'Indore'): (22.7196, 75.8577),
    ('Bihar', 'Patna'): (25.6093, 85.1376),
    ('Jharkhand', 'Ranchi'): (23.3441, 85.3096),
    ('Odisha', 'Bhubaneswar'): (20.2961, 85.8245),
    ('Assam', 'Guwahati City'): (26.1445, 91.7362),
    ('Punjab', 'Amritsar'): (31.6340, 74.8723),
    ('Punjab', 'Ludhiana'): (30.9010, 75.8573),
    ('Haryana', 'Gurugram'): (28.4595, 77.0266),
    ('Uttarakhand', 'Dehradun'): (30.3165, 78.0322),
    ('Himachal Pradesh', 'Shimla'): (31.1048, 77.1734),
    ('Jammu and Kashmir', 'Srinagar'): (34.0837, 74.7973),
    ('Jammu and Kashmir', 'Jammu'): (32.7266, 74.8570),
    ('Goa', 'North Goa'): (15.5637, 73.7509),
    ('Goa', 'South Goa'): (15.2993, 74.1240),
}


def get_district_coords(state: str, district: str) -> tuple:
    """Get coordinates for a district."""
    # Try specific district coords
    key = (state, district)
    if key in DISTRICT_COORDS:
        return DISTRICT_COORDS[key]
    
    # Try state coords with offset
    if state in STATE_COORDS:
        base_lat, base_lng = STATE_COORDS[state]
        # Add small random offset based on district name hash
        offset = hash(district) % 100 / 1000
        return (base_lat + offset, base_lng + offset)
    
    # Default to center of India
    return (20.5937, 78.9629)


def load_xgboost_model():
    """Load the trained XGBoost model and scaler."""
    import xgboost as xgb
    
    model_path = ML_DIR / "safety_classifier.json"
    scaler_path = ML_DIR / "feature_scaler.pkl"
    metadata_path = ML_DIR / "model_metadata.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load model
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return model, scaler, metadata


def load_training_data():
    """Load the training dataset with all districts."""
    training_path = ML_DIR / "training_dataset.csv"
    
    if not training_path.exists():
        raise FileNotFoundError(f"Training data not found at {training_path}")
    
    df = pd.read_csv(training_path)
    print(f"Loaded {len(df)} districts from training data")
    return df


def classify_districts(df, model, scaler, metadata):
    """Use XGBoost to classify all districts."""
    feature_cols = metadata['feature_columns']
    class_names = {0: 'Safe', 1: 'Moderate', 2: 'High'}
    zone_map = {0: 'green', 1: 'orange', 2: 'red'}
    
    # Prepare features
    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Add to dataframe
    df['predicted_class'] = predictions
    df['predicted_label'] = [class_names[p] for p in predictions]
    df['safety_zone'] = [zone_map[p] for p in predictions]
    df['confidence'] = [max(prob) for prob in probabilities]
    
    # Compute crime_index (0-100 scale based on probability of being High risk)
    df['crime_index'] = [prob[2] * 100 for prob in probabilities]  # prob of High risk
    
    print(f"\nClassification Results:")
    print(f"  Safe (Green):     {sum(predictions == 0)}")
    print(f"  Moderate (Orange): {sum(predictions == 1)}")
    print(f"  High Risk (Red):  {sum(predictions == 2)}")
    
    return df


def seed_districts_to_db(df):
    """Seed all classified districts to the database."""
    db = sessionLocal()
    added = 0
    updated = 0
    
    try:
        # Clear existing cities
        db.query(City).delete()
        db.commit()
        print("\nCleared existing cities")
        
        for _, row in df.iterrows():
            state = row.get('state', 'Unknown')
            district = row.get('district', 'Unknown')
            
            lat, lng = get_district_coords(state, district)
            
            city = City(
                name=district,
                state=state,
                latitude=lat,
                longitude=lng,
                population=int(row.get('population_lakhs', 20) * 100000) if pd.notna(row.get('population_lakhs')) else None,
                crime_index=float(row.get('crime_index', 50)),
                safety_zone=row.get('safety_zone', 'orange')
            )
            db.add(city)
            added += 1
        
        db.commit()
        print(f"Added {added} districts to database")
        
        return added
        
    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
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


def print_stats():
    """Print database statistics."""
    db = sessionLocal()
    
    try:
        total = db.query(City).count()
        green = db.query(City).filter(City.safety_zone == 'green').count()
        orange = db.query(City).filter(City.safety_zone == 'orange').count()
        red = db.query(City).filter(City.safety_zone == 'red').count()
        
        print("\n" + "=" * 50)
        print("DATABASE STATISTICS")
        print("=" * 50)
        print(f"Total locations: {total}")
        print(f"  Safe (green):     {green} ({green/total*100:.1f}%)" if total > 0 else "")
        print(f"  Moderate (orange): {orange} ({orange/total*100:.1f}%)" if total > 0 else "")
        print(f"  High Risk (red):  {red} ({red/total*100:.1f}%)" if total > 0 else "")
        
        # Sample districts by zone
        print("\nSample Districts:")
        for zone, label in [('green', 'Safe'), ('orange', 'Moderate'), ('red', 'High Risk')]:
            samples = db.query(City).filter(City.safety_zone == zone).limit(5).all()
            print(f"\n  {label}:")
            for city in samples:
                print(f"    - {city.name}, {city.state} (crime_idx: {city.crime_index:.1f})")
        
    finally:
        db.close()


def main():
    """Main entry point."""
    print("=" * 60)
    print("XGBOOST-BASED DISTRICT SEEDER")
    print("=" * 60)
    
    # Load model
    print("\nLoading XGBoost model...")
    model, scaler, metadata = load_xgboost_model()
    print(f"Model loaded with features: {metadata['feature_columns']}")
    
    # Load training data
    print("\nLoading district data...")
    df = load_training_data()
    
    # Classify all districts
    print("\nClassifying districts using XGBoost...")
    df = classify_districts(df, model, scaler, metadata)
    
    # Seed to database
    print("\nSeeding to database...")
    seed_districts_to_db(df)
    
    # Seed emergency contacts
    seed_emergency_contacts()
    
    # Print stats
    print_stats()
    
    print("\n" + "=" * 60)
    print("SEEDING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
