"""
Safety Predictor Utility for FastAPI Integration
Provides a simple interface for loading the trained model and making predictions.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print("Warning: xgboost not installed. Run: pip install xgboost")

# Default paths relative to this file
ML_DIR = Path(__file__).parent


class SafetyPredictor:
    """
    Crime Safety Prediction Interface
    
    Usage:
        predictor = SafetyPredictor()
        result = predictor.predict({
            'murder_rate': 5.0,
            'theft_rate': 50.0,
            ...
        })
    """
    
    CLASS_NAMES = {0: 'Safe', 1: 'Moderate', 2: 'High'}
    
    def __init__(self, model_path: Optional[str] = None, 
                 scaler_path: Optional[str] = None,
                 metadata_path: Optional[str] = None):
        """Initialize the predictor with trained model artifacts."""
        
        if xgb is None:
            raise ImportError("xgboost is required for predictions")
        
        self.model_path = Path(model_path) if model_path else ML_DIR / "safety_classifier.json"
        self.scaler_path = Path(scaler_path) if scaler_path else ML_DIR / "feature_scaler.pkl"
        self.metadata_path = Path(metadata_path) if metadata_path else ML_DIR / "model_metadata.json"
        
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_columns = None
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model, scaler, and metadata."""
        # Load XGBoost model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(self.model_path))
        
        # Load scaler
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")
        
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load metadata
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.feature_columns = self.metadata.get('feature_columns', [])
        else:
            # Default feature columns
            self.feature_columns = [
                'murder_rate', 'rape_rate', 'kidnapping_rate', 'robbery_rate',
                'theft_rate', 'riots_rate', 'cheating_rate',
                'crime_severity_index', 'crime_diversity_score', 'tourist_risk_score',
                'crime_rate', 'safety_score'
            ]
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict safety level from crime statistics.
        
        Args:
            features: Dictionary of crime statistics
                Required keys depend on training, but typically include:
                - murder_rate, theft_rate, robbery_rate, etc.
                
        Returns:
            Dictionary with:
                - risk_label: 'Safe', 'Moderate', or 'High'
                - risk_level: 0, 1, or 2
                - confidence: probability of predicted class
                - probabilities: dict of all class probabilities
        """
        # Prepare feature vector in correct order
        feature_vector = np.array([
            [features.get(col, 0) for col in self.feature_columns]
        ])
        
        # Scale features
        feature_scaled = self.scaler.transform(feature_vector)
        
        # Get prediction
        prediction = self.model.predict(feature_scaled)[0]
        probabilities = self.model.predict_proba(feature_scaled)[0]
        
        return {
            'risk_label': self.CLASS_NAMES[prediction],
            'risk_level': int(prediction),
            'confidence': float(max(probabilities)),
            'probabilities': {
                self.CLASS_NAMES[i]: round(float(p), 4) 
                for i, p in enumerate(probabilities)
            }
        }
    
    def predict_from_district_data(self, district_name: str, 
                                    district_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict safety level for a district using pre-computed stats.
        
        This is useful when querying from a database of district statistics.
        """
        result = self.predict(district_data)
        result['district'] = district_name
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and training information."""
        if self.metadata:
            return {
                'trained_at': self.metadata.get('trained_at'),
                'feature_columns': self.feature_columns,
                'class_names': self.CLASS_NAMES,
                'metrics': self.metadata.get('metrics', {})
            }
        return {'feature_columns': self.feature_columns}
    
    def calculate_safety_score(self, features: Dict[str, float]) -> float:
        """
        Calculate a 0-10 safety score based on prediction probabilities.
        
        Returns:
            Float from 0 (very dangerous) to 10 (very safe)
        """
        result = self.predict(features)
        probs = result['probabilities']
        
        # Weighted score: Safe=10, Moderate=5, High=0
        score = (
            probs.get('Safe', 0) * 10 +
            probs.get('Moderate', 0) * 5 +
            probs.get('High', 0) * 0
        )
        
        return round(score, 2)


# District Crime Data Cache (for quick lookups)
_district_cache = None


def load_district_features() -> Dict[str, Dict[str, float]]:
    """Load pre-computed district features for quick lookup."""
    global _district_cache
    
    if _district_cache is not None:
        return _district_cache
    
    try:
        import pandas as pd
        features_path = ML_DIR / "district_features.csv"
        
        if not features_path.exists():
            features_path = ML_DIR.parent / "crime_data" / "district_features.csv"
        
        if features_path.exists():
            df = pd.read_csv(features_path)
            _district_cache = {}
            
            for _, row in df.iterrows():
                district_key = f"{row.get('state', '')}_{row.get('district', '')}".lower()
                _district_cache[district_key] = row.to_dict()
            
            return _district_cache
    except Exception as e:
        print(f"Warning: Could not load district features: {e}")
    
    return {}


def get_safety_for_district(state: str, district: str) -> Optional[Dict[str, Any]]:
    """
    Quick lookup of safety prediction for a known district.
    
    Args:
        state: State name (e.g., "Kerala", "Maharashtra")
        district: District name (e.g., "Ernakulam", "Mumbai_City")
        
    Returns:
        Safety prediction dict or None if district not found
    """
    districts = load_district_features()
    key = f"{state}_{district}".lower().replace(' ', '_')
    
    if key in districts:
        predictor = SafetyPredictor()
        return predictor.predict(districts[key])
    
    return None


# Quick test
if __name__ == "__main__":
    print("Testing SafetyPredictor...")
    
    try:
        predictor = SafetyPredictor()
        
        # Test with sample data
        test_features = {
            'murder_rate': 5.0,
            'rape_rate': 8.0,
            'kidnapping_rate': 4.0,
            'robbery_rate': 3.0,
            'theft_rate': 50.0,
            'riots_rate': 2.0,
            'cheating_rate': 10.0,
            'crime_severity_index': 45.0,
            'crime_diversity_score': 4,
            'tourist_risk_score': 40.0,
            'crime_rate': 300.0,
            'safety_score': 6
        }
        
        result = predictor.predict(test_features)
        print(f"\nTest Prediction: {result}")
        
        safety_score = predictor.calculate_safety_score(test_features)
        print(f"Safety Score (0-10): {safety_score}")
        
    except FileNotFoundError as e:
        print(f"Model not trained yet: {e}")
        print("Run train_xgboost.py first to create the model.")
