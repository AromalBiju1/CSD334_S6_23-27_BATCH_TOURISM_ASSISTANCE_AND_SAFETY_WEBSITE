"""
Generate a detailed PDF explanation of the ML codebase.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import os

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ML_Code_Explanation.pdf")

# ─── Styles ───────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    'Title', parent=styles['Title'],
    fontSize=22, textColor=colors.HexColor('#1a1a2e'),
    spaceAfter=6, alignment=TA_CENTER
)
subtitle_style = ParagraphStyle(
    'Subtitle', parent=styles['Normal'],
    fontSize=11, textColor=colors.HexColor('#4a4a6a'),
    spaceAfter=20, alignment=TA_CENTER
)
h1_style = ParagraphStyle(
    'H1', parent=styles['Heading1'],
    fontSize=16, textColor=colors.HexColor('#16213e'),
    spaceBefore=18, spaceAfter=8,
    borderPad=4,
)
h2_style = ParagraphStyle(
    'H2', parent=styles['Heading2'],
    fontSize=12, textColor=colors.HexColor('#0f3460'),
    spaceBefore=12, spaceAfter=6,
)
body_style = ParagraphStyle(
    'Body', parent=styles['Normal'],
    fontSize=9, leading=14,
    textColor=colors.HexColor('#2d2d2d'),
    spaceAfter=4,
)
code_style = ParagraphStyle(
    'Code', parent=styles['Code'],
    fontSize=8, leading=11,
    textColor=colors.HexColor('#c7254e'),
    backColor=colors.HexColor('#f9f2f4'),
    fontName='Courier',
    leftIndent=6, rightIndent=6,
    spaceAfter=2,
)
note_style = ParagraphStyle(
    'Note', parent=styles['Normal'],
    fontSize=8.5, leading=13,
    textColor=colors.HexColor('#333333'),
    leftIndent=8,
)

# ─── Table style ──────────────────────────────────────────────────────────────
TABLE_STYLE = TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 9),
    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f4ff'), colors.white]),
    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
    ('FONTSIZE', (0, 1), (-1, -1), 8),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#c0c0d0')),
    ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ('TOPPADDING', (0, 0), (-1, -1), 5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
])

def make_table(headers, rows, col_widths):
    data = [headers] + rows
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TABLE_STYLE)
    return t

# ─── Content ──────────────────────────────────────────────────────────────────
def build_content():
    story = []
    W = A4[0] - 4*cm  # usable width

    # ── Cover ──
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph("ML Codebase — Line-by-Line Explanation", title_style))
    story.append(Paragraph("GuardMyTrip · Tourist Safety Prediction System", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0f3460')))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "This document provides a comprehensive, line-by-line explanation of every Python script "
        "in the <b>ml/</b> directory. The ML pipeline classifies Indian districts into "
        "<b>Safe</b>, <b>Moderate</b>, and <b>High Risk</b> zones using an XGBoost classifier "
        "trained on NCRB crime data.",
        body_style
    ))
    story.append(Spacer(1, 0.4*cm))

    # Table of contents
    toc = [
        ["#", "File", "Purpose"],
        ["1", "data_processor.py", "Load & clean NCRB data, compute features, auto-label districts"],
        ["2", "train_xgboost.py", "Merge data sources, engineer features, train & save XGBoost model"],
        ["3", "predictor.py", "Load trained model, expose prediction API for FastAPI backend"],
        ["4", "seed_xgboost.py", "Classify all districts, seed results into the SQL database"],
    ]
    story.append(make_table(toc[0], toc[1:], [1*cm, 5*cm, W-6*cm]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════════
    # FILE 1 — data_processor.py
    # ════════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("1. ml/data_processor.py", h1_style))
    story.append(Paragraph(
        "This is the <b>data pipeline</b> script. It reads raw NCRB (National Crime Records Bureau) "
        "CSV files, filters out non-geographic rows (railway police, CID, etc.), estimates district "
        "populations, computes per-capita crime rates, and auto-assigns risk labels using percentile "
        "thresholds. The output is <b>training_dataset.csv</b>.",
        body_style
    ))
    story.append(Spacer(1, 0.3*cm))

    dp_rows = [
        ["1–6",   '""" ... """',                          "Module docstring — states the script extracts ALL districts from NCRB data and uses crime rates to auto-label them."],
        ["8–11",  "import pandas / numpy / pathlib / logging", "Core imports: pandas for DataFrames, numpy for math, pathlib for OS-agnostic paths, logging for runtime messages."],
        ["13–14", "logging.basicConfig(...)\nlogger = ...", "Configures logging to print timestamps + severity levels. Creates a module-level logger."],
        ["17–18", "ML_DIR / DATA_DIR",                    "Dynamically resolves the script's own directory (ML_DIR) and the sibling crime_data/ folder (DATA_DIR)."],
        ["20–32", "STATE_AVG_POP_LAKHS = {...}",          "Dictionary of average district populations (in lakhs) per state, sourced from Census 2011. Used as fallback when district-level population is missing."],
        ["35–47", "NCRB_COLUMNS = {...}",                 "Maps short internal names (e.g. 'murder') to the exact verbose column headers in the NCRB CSV. Centralises the column mapping in one place."],
        ["50",    "def load_all_ncrb_districts()",        "Function: reads and cleans the raw NCRB Table 1.1 CSV."],
        ["52",    "ncrb_path = DATA_DIR / '...'",         "Builds the full path to the CSV file using pathlib (cross-platform)."],
        ["54–56", "if not ncrb_path.exists(): ...",       "Guard clause: if the file is missing, log an error and return an empty DataFrame instead of crashing."],
        ["59",    "df = pd.read_csv(ncrb_path)",          "Reads the entire CSV into a pandas DataFrame."],
        ["68–71", "exclude_patterns = [...]",             "List of keywords that identify non-geographic rows: 'Total', 'Railway', 'Crime Branch', 'CID', etc."],
        ["73",    "mask = ~df['District'].str.contains(...)", "Boolean mask using NOT (~). Keeps only rows whose District name does NOT match any exclude pattern."],
        ["75–77", "mask = mask & df[...].notna()",        "Refines the mask: also drops rows where District or State/UT is NaN (empty)."],
        ["79",    "df = df[mask].copy()",                 "Applies the mask, keeping only valid geographic district rows. .copy() prevents SettingWithCopyWarning."],
        ["83–85", "processed['state'] / ['district']",   "Creates a clean output DataFrame and populates state/district columns with stripped text."],
        ["88–93", "for name, col in NCRB_COLUMNS.items()", "Iterates the column mapping. For each crime type, converts the column to numeric (coercing errors to NaN) then fills NaN with 0."],
        ["96",    "drop_duplicates(subset=['state','district'])", "Removes duplicate state+district combinations, keeping the first occurrence."],
        ["104",   "def load_labeled_districts()",         "Loads india_district_crime_2022.csv which already has risk_label column ('Safe'/'Moderate'/'High')."],
        ["112",   "df.columns = [c.lower()...]",          "Normalises all column names to lowercase_with_underscores for consistency."],
        ["122",   "def estimate_population(df)",          "Fills missing population_lakhs values using the STATE_AVG_POP_LAKHS lookup. Defaults to 15 lakhs if state not found."],
        ["136",   "def compute_features(df)",             "Core feature engineering function. Computes all derived columns needed by the model."],
        ["144",   "pop = df['population_lakhs'].clip(lower=1)", "Clips population to minimum 1 lakh to prevent division-by-zero when computing per-capita rates."],
        ["147–153","murder_rate / rape_rate / ...",       "Per-capita crime rates: raw_count / population_lakhs * 10 (per 10 lakh). Normalises for city size."],
        ["156",   "crime_rate = total_crimes / pop",      "Overall crime rate per lakh population."],
        ["159–163","weights = {...}\ncrime_severity_index","Weighted sum of crime rates. Murder=10, Rape=8, Kidnapping=6 ... Theft=1. Captures severity, not just volume."],
        ["166–168","quantile(0.99) → normalize 0–100",   "Scales severity index to 0–100 using the 99th percentile as the max (robust to extreme outliers)."],
        ["171–172","crime_diversity_score",               "Counts how many crime types have a rate > 1 per 10 lakh. Measures breadth of criminal activity."],
        ["175–179","tourist_risk_score",                  "Sum of violent crime rates (murder, rape, robbery, kidnapping), normalised to 0–100. Specifically relevant to tourist safety."],
        ["182–186","safety_score (0–10)",                 "Inverts crime_rate: 10 − (crime_rate / max * 9). Higher = safer. Used as a human-readable score."],
        ["191",   "def assign_risk_labels(df)",           "Auto-generates the target label column for supervised learning."],
        ["199–200","p33 = quantile(0.33), p66 = quantile(0.66)", "Splits the severity distribution into thirds. Bottom third = Safe, middle = Moderate, top = High."],
        ["202–209","def classify(row)",                   "Applies the threshold logic per row. If severity ≤ p33 → 'Safe', ≤ p66 → 'Moderate', else → 'High'."],
        ["212–217","if 'risk_label' not in df.columns",  "Only generates labels for rows that don't already have one (preserves ground-truth labels from the labeled dataset)."],
        ["223",   "def create_maximum_dataset()",         "Orchestrator: calls load → merge → compute_features → assign_risk_labels → dedup → fillna."],
        ["241–249","ncrb_df.merge(labeled_df, ...)",      "Left-joins NCRB data with pre-labeled data on a composite state_district key. Prioritises labeled population/crime_rate values."],
        ["271–272","fillna(median)",                      "Final safety net: fills any remaining NaN in numeric columns with the column median."],
        ["288",   "def main()",                           "Entry point: calls create_maximum_dataset() and saves result to training_dataset.csv."],
    ]
    story.append(make_table(["Lines", "Code / Symbol", "Explanation"], dp_rows,
                             [1.5*cm, 5*cm, W-6.5*cm]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════════
    # FILE 2 — train_xgboost.py
    # ════════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("2. ml/train_xgboost.py", h1_style))
    story.append(Paragraph(
        "This is the <b>model training pipeline</b>. It loads two NCRB tables plus a labeled dataset, "
        "merges them, engineers a weighted crime score for auto-labeling, trains an XGBoost classifier, "
        "evaluates it, and saves the model artifacts (<b>.pkl</b> files) for use by the API.",
        body_style
    ))
    story.append(Spacer(1, 0.3*cm))

    tx_rows = [
        ["1–8",   '""" ... """',                          "Docstring listing all four data sources used: NCRB 1.1, NCRB 1.3, labeled 2022 data, tourist destinations."],
        ["9–17",  "import os, csv, numpy, pandas, sklearn, xgboost, joblib", "os/csv: raw file I/O. sklearn: preprocessing + evaluation. xgboost: the classifier. joblib: model serialisation."],
        ["20–22", "BASE_DIR / CRIME_DATA_DIR / MODEL_DIR","Resolves script location dynamically. CRIME_DATA_DIR points to frontend/crime_data/, MODEL_DIR to ml/models/."],
        ["25–27", "MODEL_PATH / SCALER_PATH / ENCODER_PATH", "Output file paths for the three saved artifacts: model, scaler, label encoder."],
        ["29",    "def load_ncrb_1_1()",                  "Loads NCRB Table 1.1 (IPC crimes: murder, hurt, theft, robbery, etc.)."],
        ["35–37", "skip_keywords = [...]",                "Filters out non-district rows: CID, GRP (Govt Railway Police), STF, Crime Branch, etc."],
        ["39–41", "csv.reader / next(reader)",            "Opens file with UTF-8 encoding, creates a CSV reader, skips the header row."],
        ["44",    "if len(row) < 144: continue",          "Data integrity check: skips malformed rows that don't have enough columns."],
        ["48–49", "if any(kw in district ...): continue", "Skips rows where the district name contains a skip keyword."],
        ["52–72", "record = { 'murder': float(row[3]), ...}", "Maps column indices to named fields. Uses float() with empty-string fallback to 0. Catches ValueError/IndexError silently."],
        ["77",    "pd.DataFrame(data)",                   "Converts list of dicts to a DataFrame. Each dict becomes one row."],
        ["81",    "def load_ncrb_1_3()",                  "Loads NCRB Table 1.3 (Crimes Against Women): dowry deaths, acid attacks, rape, POCSO, cyber crimes, etc."],
        ["130",   "def load_labeled_data()",              "Loads india_district_crime_2022.csv which has ground-truth Risk_Label column."],
        ["141–142","label_map = {'Safe':'green',...}",    "Converts text labels to colour codes used internally: Safe→green, Moderate→orange, High→red."],
        ["148",   "def merge_crime_data()",               "Merges all three sources into one DataFrame."],
        ["158–160","district_key = lower().strip()",      "Creates a normalised join key to handle case/spacing differences between datasets."],
        ["163",   "pd.merge(..., how='outer')",           "OUTER join: keeps all districts from both NCRB tables even if one is missing from the other."],
        ["166–167","fillna from _y columns",              "After outer join, fills missing district/state names from the duplicate _y columns."],
        ["174–175","fillna(0) on numeric cols",           "Replaces NaN in all numeric columns with 0 (crime didn't happen, not missing data)."],
        ["181",   "def create_features_and_labels()",     "Defines the feature matrix X and target y for training."],
        ["186–194","feature_cols = [...]",                "29 raw crime count columns used as model inputs (murder, rape, theft, etc.)."],
        ["200–216","crime_score = murder*15 + rape*12 + ...", "Weighted heuristic score. Weights reflect crime severity: murder=15, rape=12, dacoity=10, theft=1. Used for auto-labeling."],
        ["219–220","p33, p66 = quantile(0.33/0.66)",     "Percentile thresholds to split districts into three equal-sized groups."],
        ["222–228","def auto_label(score)",               "Assigns 'green'/'orange'/'red' based on which third of the distribution the score falls in."],
        ["240",   "def train_xgboost(df, feature_cols)",  "Core training function."],
        ["245–246","X = df[feature_cols].values, y = df['safety_zone'].values", "Extracts numpy arrays for features (X) and labels (y)."],
        ["249–250","LabelEncoder().fit_transform(y)",     "Converts string labels ('green','orange','red') to integers (0,1,2) required by XGBoost."],
        ["254–255","StandardScaler().fit_transform(X)",   "Z-score normalisation: (x − mean) / std. Ensures all features are on the same scale."],
        ["258–260","train_test_split(..., test_size=0.2, stratify=y)", "80/20 split. stratify=y ensures equal class proportions in both train and test sets."],
        ["263–273","xgb.XGBClassifier(n_estimators=150, max_depth=6, ...)", "XGBoost config: 150 trees, depth 6, learning rate 0.1, 80% row/column subsampling (reduces overfitting), multi-class softmax objective."],
        ["275",   "model.fit(X_train, y_train, eval_set=...)", "Trains the model. eval_set allows monitoring test loss during training (verbose=False suppresses output)."],
        ["278–283","accuracy_score + classification_report", "Evaluates on held-out test set. Prints per-class precision, recall, F1."],
        ["287–289","feature_importances_",                "XGBoost's built-in feature importance (gain-based). Prints top 10 most influential crime types."],
        ["293",   "def save_artifacts()",                 "Saves model, scaler, encoder, and feature column list using joblib.dump()."],
        ["312",   "def predict_and_export()",             "Runs the trained model on ALL districts and exports predictions to district_xgboost_predictions.csv for review."],
        ["323–324","predict_proba().max(axis=1)",         "Gets the highest probability across classes as the confidence score for each prediction."],
        ["339",   "def main()",                           "Orchestrates: merge → features → train → save → export."],
    ]
    story.append(make_table(["Lines", "Code / Symbol", "Explanation"], tx_rows,
                             [1.5*cm, 5.5*cm, W-7*cm]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════════
    # FILE 3 — predictor.py
    # ════════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("3. ml/predictor.py", h1_style))
    story.append(Paragraph(
        "This is the <b>inference interface</b> used by the FastAPI backend. It wraps the trained "
        "XGBoost model in a clean Python class (<b>SafetyPredictor</b>) that loads artifacts from "
        "disk and exposes simple methods for making predictions. It also provides a district-level "
        "cache for fast lookups.",
        body_style
    ))
    story.append(Spacer(1, 0.3*cm))

    pr_rows = [
        ["1–4",   '""" Safety Predictor Utility ... """', "Docstring: describes the class and shows a usage example."],
        ["6–10",  "import json, pickle, pathlib, numpy",  "Standard library imports for file handling and array math."],
        ["12–17", "try: import xgboost except ImportError", "Graceful degradation: if xgboost isn't installed, sets xgb=None and prints a warning instead of crashing at import time."],
        ["19",    "ML_DIR = Path(__file__).parent",       "Resolves the directory containing this script, used as base for all relative paths."],
        ["22",    "class SafetyPredictor:",               "OOP encapsulation: groups model, scaler, metadata, and all prediction logic into one reusable object."],
        ["35",    "CLASS_NAMES = {0:'Safe', 1:'Moderate', 2:'High'}", "Maps integer model outputs back to human-readable strings."],
        ["37–40", "def __init__(model_path, scaler_path, metadata_path)", "Constructor accepts optional custom paths; defaults to the standard filenames in ML_DIR."],
        ["49–54", "self.model = None, self.scaler = None, ...", "Initialises instance variables to None before loading. Calls _load_artifacts() to populate them."],
        ["56",    "def _load_artifacts(self)",            "Private method (leading underscore convention) that loads all three files from disk."],
        ["62–63", "xgb.XGBClassifier().load_model(path)", "Creates a blank XGBoost classifier then loads weights from the JSON file saved during training."],
        ["69–70", "pickle.load(f)",                       "Deserialises the StandardScaler object. Must use the same scaler fitted during training."],
        ["73–76", "json.load(metadata)",                  "Reads model_metadata.json to get the exact list of feature columns in the correct order."],
        ["86",    "def predict(self, features: Dict)",    "Main inference method. Accepts a dict of crime statistics, returns a structured prediction dict."],
        ["103–104","np.array([[features.get(col,0) for col in self.feature_columns]])", "Builds a 1×N feature vector in the EXACT column order used during training. Missing features default to 0."],
        ["108",   "self.scaler.transform(feature_vector)", "Applies the same z-score normalisation as during training. Critical: using transform (not fit_transform)."],
        ["111",   "self.model.predict(feature_scaled)[0]", "Returns the predicted class index (0, 1, or 2)."],
        ["112",   "self.model.predict_proba(...)[0]",     "Returns a 3-element probability array [P(Safe), P(Moderate), P(High)]."],
        ["114–122","return { risk_label, risk_level, confidence, probabilities }", "Structured response dict: human label, numeric level, max probability as confidence, full probability breakdown."],
        ["124",   "def predict_from_district_data()",     "Thin wrapper around predict() that also attaches the district name to the result dict."],
        ["146",   "def calculate_safety_score()",         "Converts probabilities to a 0–10 score: Safe×10 + Moderate×5 + High×0. Weighted average of class scores."],
        ["167",   "_district_cache = None",               "Module-level singleton cache. Loaded once on first call, reused on subsequent calls (avoids repeated disk reads)."],
        ["170",   "def load_district_features()",         "Loads district_features.csv into the cache dict keyed by 'state_district' (lowercase)."],
        ["188–190","_district_cache[key] = row.to_dict()", "Converts each CSV row to a dict and stores it. O(1) lookup by district key."],
        ["199",   "def get_safety_for_district(state, district)", "Public convenience function: looks up pre-computed stats for a district and runs predict() on them."],
        ["221",   'if __name__ == "__main__":',           "Self-test block: creates a predictor with sample data and prints the result. Only runs when script is executed directly."],
    ]
    story.append(make_table(["Lines", "Code / Symbol", "Explanation"], pr_rows,
                             [1.5*cm, 5.5*cm, W-7*cm]))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════════
    # FILE 4 — seed_xgboost.py
    # ════════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("4. ml/seed_xgboost.py", h1_style))
    story.append(Paragraph(
        "This is the <b>database seeder</b>. It bridges the ML pipeline and the application database. "
        "It loads the trained model, classifies every district in the training dataset, then writes "
        "the results (with coordinates) into the <b>City</b> table via SQLAlchemy. It also seeds "
        "national emergency contacts.",
        body_style
    ))
    story.append(Spacer(1, 0.3*cm))

    sd_rows = [
        ["1–6",   '""" XGBoost-Based District Seeder """', "Docstring: explains the script classifies all districts and seeds them into the database."],
        ["8–14",  "import sys, os, pathlib, pandas, numpy, pickle, json", "Standard imports. sys is needed to manipulate Python's module search path."],
        ["17–20", "sys.path.insert(0, str(BACKEND_DIR))", "Path hack: inserts the backend/ directory at the front of sys.path so Python can find the database/ package."],
        ["23–24", "from database.database import sessionLocal, engine\nfrom database.models import Base, City, EmergencyContact", "SQLAlchemy imports: sessionLocal creates DB sessions, City/EmergencyContact are ORM model classes."],
        ["28–68", "STATE_COORDS = {...}",                  "Hardcoded (lat, lng) for each Indian state. Used as fallback when a specific district's coordinates are unknown."],
        ["71–125","DISTRICT_COORDS = {...}",               "Hardcoded precise coordinates for ~40 major districts/cities (Mumbai, Bengaluru, Chennai, etc.) to ensure accurate map placement."],
        ["128",   "def get_district_coords(state, district)", "Coordinate resolution logic: 1) exact district match, 2) state centroid + hash-based offset (spreads dots), 3) India centre default."],
        ["139",   "offset = hash(district) % 100 / 1000", "Deterministic pseudo-random offset based on district name hash. Prevents all unknown districts in a state from stacking on the same point."],
        ["146",   "def load_xgboost_model()",             "Loads safety_classifier.json (model), feature_scaler.pkl (scaler), model_metadata.json (feature list)."],
        ["172",   "def load_training_data()",             "Reads training_dataset.csv (output of data_processor.py) into a DataFrame."],
        ["184",   "def classify_districts(df, model, scaler, metadata)", "Batch inference: scales all features at once and runs model.predict() + predict_proba() on the entire dataset."],
        ["191",   "X = df[feature_cols].copy().fillna(median)", "Prepares feature matrix; fills NaN with median to handle any missing values."],
        ["195",   "X_scaled = scaler.transform(X)",       "Applies the same scaling as during training (transform only, not fit)."],
        ["198–199","predictions = model.predict(X_scaled)", "Gets class predictions (0/1/2) for all districts simultaneously."],
        ["202–205","df['predicted_label'] / df['safety_zone']", "Adds human-readable label ('Safe') and colour code ('green') columns to the DataFrame."],
        ["208",   "df['crime_index'] = prob[2] * 100",    "Crime index = probability of being High Risk × 100. Gives a 0–100 danger score stored in the DB."],
        ["218",   "def seed_districts_to_db(df)",         "Writes classified districts to the database."],
        ["220",   "db = sessionLocal()",                  "Opens a new SQLAlchemy database session."],
        ["226–227","db.query(City).delete() / db.commit()", "⚠️ DESTRUCTIVE: deletes ALL existing City rows before re-seeding. Ensures a clean slate."],
        ["230",   "for _, row in df.iterrows():",         "Iterates every district row in the classified DataFrame."],
        ["236–244","city = City(name=..., latitude=..., crime_index=..., safety_zone=...)", "Creates an ORM City object. population is converted from lakhs to absolute number (×100,000)."],
        ["245",   "db.add(city)",                         "Stages the City object for insertion (not yet committed to DB)."],
        ["248",   "db.commit()",                          "Commits all staged inserts in one transaction (efficient batch write)."],
        ["253–258","except Exception: db.rollback()",     "On any error, rolls back the entire transaction to keep the DB consistent."],
        ["259",   "finally: db.close()",                  "Always closes the session, even if an exception occurred."],
        ["263",   "def seed_emergency_contacts()",        "Inserts national emergency numbers (Police 100, Ambulance 102, Women Helpline 1091, etc.) into EmergencyContact table."],
        ["302",   "def print_stats()",                    "Queries the DB and prints a summary: total cities, count per safety zone, and 5 sample districts per zone."],
        ["332",   "def main()",                           "Full pipeline: load model → load data → classify → seed cities → seed contacts → print stats."],
    ]
    story.append(make_table(["Lines", "Code / Symbol", "Explanation"], sd_rows,
                             [1.5*cm, 5.5*cm, W-7*cm]))
    story.append(Spacer(1, 0.5*cm))

    # ── Pipeline summary ──
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#0f3460')))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("End-to-End Pipeline Summary", h2_style))
    pipeline = [
        ["Step", "Script", "Input", "Output"],
        ["1 — Data Prep",   "data_processor.py",  "NCRB_District_Table_1.1.csv\nindia_district_crime_2022.csv", "training_dataset.csv"],
        ["2 — Training",    "train_xgboost.py",   "NCRB 1.1 + 1.3 CSVs",         "safety_classifier.pkl\nscaler.pkl\nlabel_encoder.pkl"],
        ["3 — Inference",   "predictor.py",       "safety_classifier.json\nfeature_scaler.pkl", "JSON prediction dict\n(risk_label, confidence, score)"],
        ["4 — DB Seeding",  "seed_xgboost.py",    "training_dataset.csv\nsafety_classifier.json", "City rows in SQL DB\nEmergencyContact rows"],
    ]
    story.append(make_table(pipeline[0], pipeline[1:], [3*cm, 4*cm, 5*cm, W-12*cm]))

    return story


# ─── Build PDF ────────────────────────────────────────────────────────────────
def main():
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="ML Codebase Explanation — GuardMyTrip",
        author="Antigravity AI",
    )
    story = build_content()
    doc.build(story)
    print(f"PDF saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
