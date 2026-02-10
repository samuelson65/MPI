import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. ROBUST DATA GENERATION (Training Phase)
# ==========================================
np.random.seed(42)

# Create "Normal" Patterns for the AI to learn
# Pattern A: Diabetes (E11) -> Insulin (J1815)
df_diabetes = pd.DataFrame({
    'Raw_Diagnosis': np.random.choice(['E11.9', 'E11.2', 'E11.65'], 2000),
    'J_Code': ['J1815'] * 2000,
    'Units': np.random.normal(30, 5, 2000)
})

# Pattern B: Asthma (J45) -> Albuterol (J7613)
df_asthma = pd.DataFrame({
    'Raw_Diagnosis': np.random.choice(['J45.909', 'J45.4'], 2000),
    'J_Code': ['J7613'] * 2000,
    'Units': np.random.normal(10, 2, 2000)
})

# Combine into training set
df_train = pd.concat([df_diabetes, df_asthma]).sample(frac=1).reset_index(drop=True)

# --- PREPROCESSING HELPER ---
def clean_diagnosis(df, col_name):
    # Ensure string, strip whitespace, take first 3 chars, uppercase
    df['Diag_Category'] = df[col_name].astype(str).str.strip().str[:3].str.upper()
    return df

df_train = clean_diagnosis(df_train, 'Raw_Diagnosis')

# --- ROBUST ENCODING ---
# Handles "New" codes without crashing
class SafeEncoder(LabelEncoder):
    def transform(self, y):
        # Check if the code is known. If yes, encode. If no, assign -1.
        return np.array([super().transform([x])[0] if x in self.classes_ else -1 for x in y])

le_diag = SafeEncoder()
le_jcode = SafeEncoder()

df_train['Diag_Encoded'] = le_diag.fit_transform(df_train['Diag_Category'])
df_train['JCode_Encoded'] = le_jcode.fit_transform(df_train['J_Code'])

features = ['Diag_Encoded', 'JCode_Encoded', 'Units']

# --- TRAIN MODEL ---
iso_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
iso_forest.fit(df_train[features])
print("--- AI Trained on Normal Patterns ---")

# ==========================================
# 2. COMPLEX INPUT (The "Unequal Lists" Scenario)
# ==========================================
# Scenario: 
# - 2 Diagnoses (Diabetes, Asthma)
# - 3 Procedures (Insulin, Chemo, Albuterol)
# - 3 Units (Matches Procedures)
# NOTE: Diagnoses length (2) != J_Codes length (3). This usually breaks Pandas.

raw_data = [
    {
        'Claim_ID': 'CLM_001',
        'Diagnoses': ['E11.9', 'J45.909'], 
        'J_Codes':   ['J1815', 'J9000', 'J7613'], 
        'Units':     [30, 50, 600] 
    }
]

df_input = pd.DataFrame(raw_data)

# ==========================================
# 3. THE ROBUST FIX (Step-by-Step Explosion)
# ==========================================

# STEP A: Explode the LINE ITEMS first (J_Codes + Units)
# These lists MUST be the same length. We explode them together.
# 'Diagnoses' remains a list on every row for now.
df_lines = df_input.explode(['J_Codes', 'Units']).reset_index(drop=True)

# STEP B: Explode the CONTEXT (Diagnoses)
# Now we explode Diagnoses. This creates a "Cross Join" effect.
# (Row 1: Insulin) x (Diag A, Diag B) = 2 Rows to check.
df_pairs = df_lines.explode('Diagnoses').reset_index(drop=True)

# Rename for processing
df_pairs = df_pairs.rename(columns={'Diagnoses': 'Raw_Diagnosis', 'J_Codes': 'J_Code'})

# Ensure data types are correct after explosion
df_pairs['Units'] = pd.to_numeric(df_pairs['Units'])

# ==========================================
# 4. PREDICT & AGGREGATE
# ==========================================

# Preprocess New Data
df_pairs = clean_diagnosis(df_pairs, 'Raw_Diagnosis')
df_pairs['Diag_Encoded'] = le_diag.transform(df_pairs['Diag_Category'])
df_pairs['JCode_Encoded'] = le_jcode.transform(df_pairs['J_Code'])

# AI Prediction
# Returns: Score > 0 (Normal), Score < 0 (Anomaly)
df_pairs['Anomaly_Score'] = iso_forest.decision_function(df_pairs[features])

# AGGREGATION LOGIC:
# For every Line Item (J_Code + Units), we take the BEST score found.
# If *any* diagnosis makes the drug valid, the Max Score will be high.
final_report = df_pairs.groupby(['Claim_ID', 'J_Code', 'Units']).agg(
    Best_Score=('Anomaly_Score', 'max')
).reset_index()

# ==========================================
# 5. FINAL REASONING LOGIC
# ==========================================
def generate_decision(row):
    if row['Best_Score'] > 0:
        return "APPROVED"
    
    # If flagged, guess the reason based on the input values
    if row['Units'] > 100:
        return f"FLAG: Excessive Units ({row['Units']})"
    
    return "FLAG: Clinical Mismatch (Orphan Drug)"

final_report['Decision'] = final_report.apply(generate_decision, axis=1)

print("\n--- FINAL ROBUST AUDIT ---")
print(final_report.to_string(index=False))
