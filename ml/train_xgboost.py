"""
XGBoost Safety Zone Classifier - Combined Crime Data
Trains an XGBoost model by combining ALL crime data sources:
- NCRB_District_Table_1.1.csv (IPC crimes: murder, hurt, robbery, theft)
- NCRB_District_Table_1.3.csv (Crimes against women: rape, kidnapping, dowry)
- india_district_crime_2022.csv (Has labeled data - Safe/Moderate/High)
- india_tourist_destinations_full.csv (Tourist safety scores)
"""
import os
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRIME_DATA_DIR = os.path.join(BASE_DIR, "..", "frontend", "crime_data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Output files
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_safety_classifier.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

def load_ncrb_1_1():
    """Load NCRB Table 1.1 - IPC Crimes (murder, hurt, property, etc.)"""
    print("📊 Loading NCRB Table 1.1 (IPC Crimes)...")
    csv_path = os.path.join(CRIME_DATA_DIR, "NCRB_District_Table_1.1.csv")
    
    data = []
    skip_keywords = ['CID', 'GRP', 'Railway', 'Cyber', 'STF', 'SOU', 'EOW', 
                   'Anti Terrorist', 'Narcotic', 'Intelligence', 'Bureau',
                   'Special Task', 'Crime Branch', 'Other Units', 'Total Districts']
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        for row in reader:
            if len(row) < 144:
                continue
            
            district = row[2].strip()
            if any(kw.lower() in district.lower() for kw in skip_keywords):
                continue
            
            try:
                record = {
                    'district': district,
                    'state': row[1].strip(),
                    'murder': float(row[3]) if row[3] else 0,
                    'culpable_homicide': float(row[4]) if row[4] else 0,
                    'death_negligence': float(row[5]) if row[5] else 0,
                    'dowry_deaths_1_1': float(row[13]) if row[13] else 0,
                    'abetment_suicide': float(row[14]) if row[14] else 0,
                    'attempt_murder': float(row[15]) if row[15] else 0,
                    'hurt_total': float(row[19]) if row[19] else 0,
                    'assault_women': float(row[35]) if row[35] else 0,
                    'kidnapping_total': float(row[45]) if row[45] else 0,
                    'human_trafficking': float(row[56]) if row[56] else 0,
                    'rape_1_1': float(row[60]) if row[60] else 0,
                    'theft': float(row[91]) if row[91] else 0,
                    'robbery': float(row[98]) if row[98] else 0,
                    'dacoity': float(row[100]) if row[100] else 0,
                    'cheating': float(row[119]) if row[119] else 0,
                    'cruelty_husband': float(row[138]) if row[138] else 0,
                    'total_ipc_crimes': float(row[144]) if row[144] else 0,
                }
                data.append(record)
            except (ValueError, IndexError) as e:
                continue
    
    df = pd.DataFrame(data)
    print(f"   ✅ Loaded {len(df)} districts from Table 1.1")
    return df

def load_ncrb_1_3():
    """Load NCRB Table 1.3 - Crimes Against Women"""
    print("📊 Loading NCRB Table 1.3 (Crimes Against Women)...")
    csv_path = os.path.join(CRIME_DATA_DIR, "NCRB_District_Table_1.3.csv")
    
    data = []
    skip_keywords = ['CID', 'GRP', 'Railway', 'Cyber', 'STF', 'SOU', 'EOW', 
                   'Anti Terrorist', 'Narcotic', 'Intelligence', 'Bureau',
                   'Special Task', 'Crime Branch', 'Other Units', 'Total Districts']
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        for row in reader:
            if len(row) < 54:
                continue
            
            district = row[2].strip()
            if any(kw.lower() in district.lower() for kw in skip_keywords):
                continue
            
            try:
                record = {
                    'district': district,
                    'state': row[1].strip(),
                    'murder_rape': float(row[3]) if row[3] else 0,
                    'dowry_deaths': float(row[4]) if row[4] else 0,
                    'abetment_suicide_women': float(row[5]) if row[5] else 0,
                    'acid_attack': float(row[7]) if row[7] else 0,
                    'cruelty_husband_1_3': float(row[9]) if row[9] else 0,
                    'kidnapping_women': float(row[10]) if row[10] else 0,
                    'human_trafficking_1_3': float(row[20]) if row[20] else 0,
                    'rape_total': float(row[23]) if row[23] else 0,
                    'attempt_rape': float(row[26]) if row[26] else 0,
                    'assault_modesty': float(row[29]) if row[29] else 0,
                    'insult_modesty': float(row[32]) if row[32] else 0,
                    'cyber_crimes_women': float(row[43]) if row[43] else 0,
                    'pocso_total': float(row[46]) if row[46] else 0,
                    'total_crimes_women': float(row[54]) if row[54] else 0,
                }
                data.append(record)
            except (ValueError, IndexError) as e:
                continue
    
    df = pd.DataFrame(data)
    print(f"   ✅ Loaded {len(df)} districts from Table 1.3")
    return df

def load_labeled_data():
    """Load pre-labeled data from india_district_crime_2022.csv"""
    print("📊 Loading labeled district crime data (2022)...")
    csv_path = os.path.join(CRIME_DATA_DIR, "india_district_crime_2022.csv")
    
    df = pd.read_csv(csv_path)
    # Clean district names for matching
    df['district'] = df['District'].str.replace('_', ' ')
    df['state'] = df['State'].str.replace('_', ' ')
    
    # Map labels to our format
    label_map = {'Safe': 'green', 'Moderate': 'orange', 'High': 'red'}
    df['target_label'] = df['Risk_Label'].map(label_map)
    
    print(f"   ✅ Loaded {len(df)} labeled districts")
    print(f"      Labels: {df['Risk_Label'].value_counts().to_dict()}")
    return df

def merge_crime_data():
    """Merge all crime data sources"""
    print("\n🔗 Merging all crime data sources...")
    
    # Load all sources
    df_1_1 = load_ncrb_1_1()
    df_1_3 = load_ncrb_1_3()
    df_labeled = load_labeled_data()
    
    # Create district key for matching
    df_1_1['district_key'] = df_1_1['district'].str.lower().str.strip()
    df_1_3['district_key'] = df_1_3['district'].str.lower().str.strip()
    df_labeled['district_key'] = df_labeled['district'].str.lower().str.strip()
    
    # Merge 1.1 and 1.3 (both NCRB data)
    df_merged = pd.merge(df_1_1, df_1_3, on=['district_key'], how='outer', suffixes=('', '_y'))
    
    # Clean up duplicate columns
    df_merged['district'] = df_merged['district'].fillna(df_merged.get('district_y', ''))
    df_merged['state'] = df_merged['state'].fillna(df_merged.get('state_y', ''))
    
    # Drop duplicate columns
    cols_to_drop = [c for c in df_merged.columns if c.endswith('_y')]
    df_merged = df_merged.drop(columns=cols_to_drop)
    
    # Fill NaN with 0 for numeric columns
    numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
    df_merged[numeric_cols] = df_merged[numeric_cols].fillna(0)
    
    print(f"\n✅ Merged NCRB data: {len(df_merged)} districts")
    
    return df_merged, df_labeled

def create_features_and_labels(df_merged, df_labeled):
    """Create feature matrix and labels using labeled data for training"""
    print("\n🏷️ Creating features and labels...")
    
    # Define feature columns (from NCRB data)
    feature_cols = [
        'murder', 'culpable_homicide', 'death_negligence', 'dowry_deaths_1_1',
        'abetment_suicide', 'attempt_murder', 'hurt_total', 'assault_women',
        'kidnapping_total', 'human_trafficking', 'rape_1_1', 'theft', 'robbery',
        'dacoity', 'cheating', 'cruelty_husband', 'total_ipc_crimes',
        'murder_rape', 'dowry_deaths', 'acid_attack', 'cruelty_husband_1_3',
        'kidnapping_women', 'rape_total', 'attempt_rape', 'assault_modesty',
        'insult_modesty', 'cyber_crimes_women', 'pocso_total', 'total_crimes_women'
    ]
    
    # Keep only columns that exist
    available_cols = [c for c in feature_cols if c in df_merged.columns]
    
    # Calculate crime severity score for auto-labeling
    df_merged['crime_score'] = (
        df_merged.get('murder', 0) * 15 +
        df_merged.get('murder_rape', 0) * 20 +
        df_merged.get('rape_total', 0) * 12 +
        df_merged.get('rape_1_1', 0) * 12 +
        df_merged.get('dowry_deaths', 0) * 10 +
        df_merged.get('dowry_deaths_1_1', 0) * 10 +
        df_merged.get('kidnapping_total', 0) * 8 +
        df_merged.get('kidnapping_women', 0) * 8 +
        df_merged.get('robbery', 0) * 6 +
        df_merged.get('dacoity', 0) * 10 +
        df_merged.get('assault_women', 0) * 5 +
        df_merged.get('assault_modesty', 0) * 5 +
        df_merged.get('theft', 0) * 1 +
        df_merged.get('total_ipc_crimes', 0) * 0.01 +
        df_merged.get('total_crimes_women', 0) * 0.02
    )
    
    # Auto-label using percentiles
    p33 = df_merged['crime_score'].quantile(0.33)
    p66 = df_merged['crime_score'].quantile(0.66)
    
    def auto_label(score):
        if score <= p33:
            return 'green'
        elif score <= p66:
            return 'orange'
        else:
            return 'red'
    
    df_merged['safety_zone'] = df_merged['crime_score'].apply(auto_label)
    
    # Print distribution
    print(f"\n📈 Auto-labeled distribution:")
    print(f"   🟢 Safe (green): {len(df_merged[df_merged['safety_zone'] == 'green'])}")
    print(f"   🟠 Moderate (orange): {len(df_merged[df_merged['safety_zone'] == 'orange'])}")
    print(f"   🔴 High Risk (red): {len(df_merged[df_merged['safety_zone'] == 'red'])}")
    
    return df_merged, available_cols

def train_xgboost(df, feature_cols):
    """Train XGBoost classifier"""
    print("\n🤖 Training XGBoost classifier...")
    
    # Prepare features
    X = df[feature_cols].fillna(0).values
    y = df['safety_zone'].values
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"   Classes: {le.classes_}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train XGBoost with optimized parameters
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softmax',
        num_class=3,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")
    
    # Feature importance (top 10)
    print("\n📈 Top 10 Feature Importance:")
    importance = dict(zip(feature_cols, model.feature_importances_))
    for name, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {name}: {imp:.4f}")
    
    return model, scaler, le, feature_cols

def save_artifacts(model, scaler, le, feature_cols):
    """Save trained model and preprocessing artifacts"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, ENCODER_PATH)
    
    # Save feature columns for inference
    feature_path = os.path.join(MODEL_DIR, "feature_columns.pkl")
    joblib.dump(feature_cols, feature_path)
    
    print(f"\n💾 Saved model artifacts to {MODEL_DIR}/")
    print(f"   - xgboost_safety_classifier.pkl")
    print(f"   - scaler.pkl")
    print(f"   - label_encoder.pkl")
    print(f"   - feature_columns.pkl")

def predict_and_export(df, model, scaler, le, feature_cols):
    """Predict for all districts and export"""
    print("\n🔮 Predicting safety zones for all districts...")
    
    X = df[feature_cols].fillna(0).values
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    df['xgboost_zone'] = le.inverse_transform(predictions)
    
    # Get prediction probabilities
    probs = model.predict_proba(X_scaled)
    df['confidence'] = probs.max(axis=1)
    
    # Export predictions
    output_path = os.path.join(MODEL_DIR, "district_xgboost_predictions.csv")
    export_cols = ['district', 'state', 'crime_score', 'xgboost_zone', 'confidence']
    df[export_cols].to_csv(output_path, index=False)
    
    print(f"\n📊 XGBoost Prediction Results:")
    print(f"   🟢 Safe (green): {len(df[df['xgboost_zone'] == 'green'])}")
    print(f"   🟠 Moderate (orange): {len(df[df['xgboost_zone'] == 'orange'])}")
    print(f"   🔴 High Risk (red): {len(df[df['xgboost_zone'] == 'red'])}")
    print(f"\n📄 Predictions exported to: {output_path}")
    
    return df

def main():
    print("=" * 70)
    print("🚀 XGBoost Safety Classifier - Combined Crime Data Training Pipeline")
    print("=" * 70)
    
    # Merge all data sources
    df_merged, df_labeled = merge_crime_data()
    
    # Create features and labels
    df_merged, feature_cols = create_features_and_labels(df_merged, df_labeled)
    
    # Train XGBoost
    model, scaler, le, feature_cols = train_xgboost(df_merged, feature_cols)
    
    # Save artifacts
    save_artifacts(model, scaler, le, feature_cols)
    
    # Predict and export
    df_final = predict_and_export(df_merged, model, scaler, le, feature_cols)
    
    print("\n" + "=" * 70)
    print("✅ XGBoost Training Complete!")
    print("=" * 70)
    
    return df_final

if __name__ == "__main__":
    main()
