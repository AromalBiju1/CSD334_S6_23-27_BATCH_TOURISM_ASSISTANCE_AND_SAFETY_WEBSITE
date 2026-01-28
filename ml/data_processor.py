"""
Maximum Coverage Crime Data Processor
=====================================
Extracts ALL districts from NCRB data (900+) and computes features.
Uses crime rates to auto-label districts for training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
ML_DIR = Path(__file__).parent
DATA_DIR = ML_DIR.parent / "crime_data"

# Average district populations by state (in lakhs) - from Census 2011
STATE_AVG_POP_LAKHS = {
    'Andhra Pradesh': 35.0, 'Arunachal Pradesh': 5.0, 'Assam': 20.0, 'Bihar': 28.0,
    'Chhattisgarh': 18.0, 'Goa': 7.0, 'Gujarat': 20.0, 'Haryana': 22.0,
    'Himachal Pradesh': 6.0, 'Jharkhand': 26.0, 'Karnataka': 25.0, 'Kerala': 24.0,
    'Madhya Pradesh': 27.0, 'Maharashtra': 32.0, 'Manipur': 3.0, 'Meghalaya': 6.0,
    'Mizoram': 3.0, 'Nagaland': 2.5, 'Odisha': 14.0, 'Punjab': 20.0,
    'Rajasthan': 22.0, 'Sikkim': 2.0, 'Tamil Nadu': 25.0, 'Telangana': 24.0,
    'Tripura': 10.0, 'Uttar Pradesh': 28.0, 'Uttarakhand': 8.0, 'West Bengal': 24.0,
    'A & N Islands': 3.0, 'Chandigarh': 10.0, 'D & N Haveli': 3.0, 'Daman & Diu': 2.0,
    'Delhi': 140.0, 'Jammu & Kashmir': 12.0, 'Ladakh': 3.0, 'Lakshadweep': 0.6,
    'Puducherry': 12.0,
}

# NCRB column mappings (simplified)
NCRB_COLUMNS = {
    'murder': 'Offences affecting the Human Body - Murder (Sec.302 IPC) - Col. ( 3)',
    'rape': 'Offences affecting the Human Body - Rape (Sec.376 IPC) - Col. ( 60)',
    'kidnapping': 'Offences affecting the Human Body - Kidnapping and Abduction - Kidnapping and Abduction (Total) (Col.46+ Col.49 to Col.55) - Col. ( 45)',
    'robbery': 'Offences against Property - Robbery (Sec.392/394/397 IPC) - Col. ( 98)',
    'theft': 'Offences against Property - Theft (Section 379 IPC) - Theft (Total) (Col.92+Col.93) - Col. ( 91)',
    'riots': 'Offences against Public Tranquillity - Rioting (Sec.1470151 IPC) - Rioting (Total) (Col.70 to Col.85) - Col. ( 69)',
    'cheating': 'Offences Relating to Documents & Property Marks - Forgery, Cheating & Fraud - Cheating (Sec.420 IPC) - - Col. ( 119)',
    'total_crimes': 'Total Cognizable IPC crimes - Col. ( 144)',
    'hurt': 'Offences affecting the Human Body - Hurt - Hurt (Total) (Col.20 + Col.26) - Col. ( 19)',
    'assault_women': 'Offences affecting the Human Body - Assault on Women with Intent to Outrage her Modesty - Assault on Women with Intent to Outrage her Modesty (Sec.354 IPC) (Total) (Col.36+Col.37+Col42 to 44) - Col. ( 35)',
    'dowry_deaths': 'Offences affecting the Human Body - Dowry Deaths (Sec.304-B IPC) - Col. ( 13)',
}


def load_all_ncrb_districts() -> pd.DataFrame:
    """Load ALL districts from NCRB data."""
    ncrb_path = DATA_DIR / "NCRB_District_Table_1.1.csv"
    
    if not ncrb_path.exists():
        logger.error(f"NCRB data not found at {ncrb_path}")
        return pd.DataFrame()
    
    logger.info(f"Loading NCRB data from {ncrb_path}")
    df = pd.read_csv(ncrb_path)
    logger.info(f"Raw rows: {len(df)}")
    
    # Filter out:
    # 1. Total/Summary rows
    # 2. Railway districts (not geographic)
    # 3. Crime Branch (not a district)
    # 4. Rows where District == State (state totals)
    
    exclude_patterns = [
        'Total', 'Railway', 'Crime Branch', 'CID', 'GRP', 
        'Metro', 'Commissionerate', 'Range', 'Division'
    ]
    
    mask = ~df['District'].str.contains('|'.join(exclude_patterns), case=False, na=False)
    # Also exclude if District is NaN
    mask = mask & df['District'].notna()
    # Exclude if State/UT is NaN
    mask = mask & df['State/UT'].notna()
    
    df = df[mask].copy()
    logger.info(f"After filtering: {len(df)} districts")
    
    # Build processed dataframe
    processed = pd.DataFrame()
    processed['state'] = df['State/UT'].str.strip()
    processed['district'] = df['District'].str.strip()
    
    # Extract crime counts
    for name, col in NCRB_COLUMNS.items():
        if col in df.columns:
            processed[name] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            processed[name] = 0
            logger.warning(f"Column not found: {name}")
    
    # Remove duplicates
    processed = processed.drop_duplicates(subset=['state', 'district'], keep='first')
    
    logger.info(f"Loaded {len(processed)} unique districts from NCRB")
    logger.info(f"States covered: {processed['state'].nunique()}")
    
    return processed


def load_labeled_districts() -> pd.DataFrame:
    """Load districts that already have risk labels."""
    path = DATA_DIR / "india_district_crime_2022.csv"
    
    if not path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    
    # Standardize names
    df['state'] = df['state'].str.replace('_', ' ').str.strip()
    df['district'] = df['district'].str.replace('_', ' ').str.strip()
    
    logger.info(f"Loaded {len(df)} pre-labeled districts")
    return df[['state', 'district', 'risk_label', 'population_lakhs', 'crime_rate']]


def estimate_population(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate population for districts based on state averages."""
    def get_pop(row):
        state = row['state']
        # Try exact match
        for s, pop in STATE_AVG_POP_LAKHS.items():
            if s.lower() in state.lower() or state.lower() in s.lower():
                return pop
        return 15.0  # Default
    
    df['population_lakhs'] = df.apply(get_pop, axis=1)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features needed for ML training."""
    logger.info("Computing features...")
    
    # Estimate population if not present
    if 'population_lakhs' not in df.columns:
        df = estimate_population(df)
    
    pop = df['population_lakhs'].clip(lower=1)  # Avoid division by zero
    
    # Per-capita rates
    df['murder_rate'] = df['murder'] / pop * 10  # per 10 lakh
    df['rape_rate'] = df['rape'] / pop * 10
    df['kidnapping_rate'] = df['kidnapping'] / pop * 10
    df['robbery_rate'] = df['robbery'] / pop * 10
    df['theft_rate'] = df['theft'] / pop * 10
    df['riots_rate'] = df['riots'] / pop * 10
    df['cheating_rate'] = df['cheating'] / pop * 10
    
    # Overall crime rate
    df['crime_rate'] = df['total_crimes'] / pop
    
    # Crime severity index (weighted)
    weights = {
        'murder_rate': 10, 'rape_rate': 8, 'kidnapping_rate': 6,
        'robbery_rate': 5, 'riots_rate': 3, 'cheating_rate': 2, 'theft_rate': 1
    }
    df['crime_severity_index'] = sum(df[col] * w for col, w in weights.items())
    
    # Normalize to 0-100
    max_severity = df['crime_severity_index'].quantile(0.99)
    if max_severity > 0:
        df['crime_severity_index'] = (df['crime_severity_index'] / max_severity * 100).clip(0, 100)
    
    # Crime diversity (count of significant crime types)
    rate_cols = ['murder_rate', 'rape_rate', 'kidnapping_rate', 'robbery_rate', 'theft_rate', 'riots_rate']
    df['crime_diversity_score'] = (df[rate_cols] > 1).sum(axis=1)
    
    # Tourist risk score (violent crimes)
    violent_cols = ['murder_rate', 'rape_rate', 'robbery_rate', 'kidnapping_rate']
    df['tourist_risk_score'] = df[violent_cols].sum(axis=1)
    max_tourist = df['tourist_risk_score'].quantile(0.99)
    if max_tourist > 0:
        df['tourist_risk_score'] = (df['tourist_risk_score'] / max_tourist * 100).clip(0, 100)
    
    # Safety score (1-10, inverse of crime)
    max_crime = df['crime_rate'].quantile(0.99)
    if max_crime > 0:
        df['safety_score'] = 10 - (df['crime_rate'] / max_crime * 9).clip(0, 9)
    else:
        df['safety_score'] = 5
    
    return df


def assign_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-assign risk labels based on crime metrics."""
    logger.info("Assigning risk labels...")
    
    # Use crime_severity_index as primary metric
    severity = df['crime_severity_index']
    
    # Compute percentiles for thresholds
    p33 = severity.quantile(0.33)
    p66 = severity.quantile(0.66)
    
    def classify(row):
        s = row['crime_severity_index']
        if s <= p33:
            return 'Safe'
        elif s <= p66:
            return 'Moderate'
        else:
            return 'High'
    
    # Only assign if not already labeled
    if 'risk_label' not in df.columns:
        df['risk_label'] = df.apply(classify, axis=1)
    else:
        # Fill missing labels
        mask = df['risk_label'].isna() | (df['risk_label'] == '')
        df.loc[mask, 'risk_label'] = df[mask].apply(classify, axis=1)
    
    logger.info(f"Label distribution: {df['risk_label'].value_counts().to_dict()}")
    return df


def create_maximum_dataset() -> pd.DataFrame:
    """Create the maximum coverage training dataset."""
    logger.info("=" * 60)
    logger.info("CREATING MAXIMUM COVERAGE DATASET")
    logger.info("=" * 60)
    
    # Load ALL NCRB districts
    ncrb_df = load_all_ncrb_districts()
    
    # Load pre-labeled districts
    labeled_df = load_labeled_districts()
    
    if len(ncrb_df) == 0:
        logger.error("No NCRB data loaded!")
        return pd.DataFrame()
    
    # Merge labels where available
    if len(labeled_df) > 0:
        ncrb_df['merge_key'] = ncrb_df['state'].str.lower() + '_' + ncrb_df['district'].str.lower()
        labeled_df['merge_key'] = labeled_df['state'].str.lower() + '_' + labeled_df['district'].str.lower()
        
        ncrb_df = ncrb_df.merge(
            labeled_df[['merge_key', 'risk_label', 'population_lakhs', 'crime_rate']],
            on='merge_key',
            how='left',
            suffixes=('', '_labeled')
        )
        ncrb_df = ncrb_df.drop(columns=['merge_key'], errors='ignore')
        
        # Use labeled values where available
        if 'population_lakhs_labeled' in ncrb_df.columns:
            mask = ncrb_df['population_lakhs_labeled'].notna()
            ncrb_df.loc[mask, 'population_lakhs'] = ncrb_df.loc[mask, 'population_lakhs_labeled']
            ncrb_df = ncrb_df.drop(columns=['population_lakhs_labeled'], errors='ignore')
        
        if 'crime_rate_labeled' in ncrb_df.columns:
            ncrb_df = ncrb_df.drop(columns=['crime_rate_labeled'], errors='ignore')
    
    # Compute features
    df = compute_features(ncrb_df)
    
    # Assign labels where missing
    df = assign_risk_labels(df)
    
    # Clean up
    df = df.drop_duplicates(subset=['state', 'district'], keep='first')
    
    # Fill any remaining NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Final stats
    logger.info("=" * 60)
    logger.info("FINAL DATASET STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total districts: {len(df)}")
    logger.info(f"States covered: {df['state'].nunique()}")
    logger.info(f"Risk distribution:")
    logger.info(f"  Safe:     {sum(df['risk_label'] == 'Safe')}")
    logger.info(f"  Moderate: {sum(df['risk_label'] == 'Moderate')}")
    logger.info(f"  High:     {sum(df['risk_label'] == 'High')}")
    
    return df


def main():
    """Main entry point."""
    df = create_maximum_dataset()
    
    if len(df) > 0:
        output_path = ML_DIR / "training_dataset.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} districts to {output_path}")
        
        # Print sample
        print("\nSample Data (first 20 districts):")
        print(df[['state', 'district', 'crime_rate', 'crime_severity_index', 'risk_label']].head(20).to_string())
    else:
        print("Failed to create dataset!")


if __name__ == "__main__":
    main()
