import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. SETUP TRAIN DATA (The "Knowledge Base")
# ==========================================
np.random.seed(42)

# Create "Normal" Patterns
# Pattern A: Diabetes (E11) -> Insulin (J1815)
df_diabetes = pd.DataFrame({
    'Raw_Diagnosis': np.random.choice(['E11.9', 'E11.2'], 2000),
    'J_Code': ['J1815'] * 2000,
    'Units': np.random.normal(30, 5, 2000)
})

# Pattern B: Asthma (J45) -> Albuterol (J7613)
df_asthma = pd.DataFrame({
    'Raw_Diagnosis': np.random.choice(['J45.909', 'J45.4'], 2000),
    'J_Code': ['J7613'] * 2000,
    'Units': np.random.normal(10, 2, 2000)
})

df_train = pd.concat([df_diabetes, df_asthma]).sample(frac=1).reset_index(drop=True)

# --- PREPROCESSING ---
def clean_diagnosis(df, col_name):
    df['Diag_Category'] = df[col_name].astype(str).str.strip().str[:3].str.upper()
    return df

df_train = clean_diagnosis(df_train, 'Raw_Diagnosis')

# --- ENCODING ---
class SafeEncoder(LabelEncoder):
    def transform(self, y):
        return np.array([super().transform([x])[0] if x in self.classes_ else -1 for x in y])

le_diag = SafeEncoder()
le_jcode = SafeEncoder()

df_train['Diag_Encoded'] = le_diag.fit_transform(df_train['Diag_Category'])
df_train['JCode_Encoded'] = le_jcode.fit_transform(df_train['J_Code'])

features = ['Diag_Encoded', 'JCode_Encoded', 'Units']

# --- TRAIN MODEL ---
iso_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
iso_forest.fit(df_train[features])
print("--- AI Model Trained ---")

# ==========================================
# 2. COMPLEX INPUT (Uneven Lists)
# ==========================================
# Scenario: 2 Diagnoses, 3 Procedures.
# This caused the crash before because indices got duplicated.

raw_data = [
    {
        'Claim_ID': 'CLM_CRASH_FIXED',
        'Diagnoses': ['E11.9', 'J45.909'],  # Length 2
        'J_Codes':   ['J1815', 'J9000', 'J7613'], # Length 3
        'Units':     [30, 50, 600] # Length 3
    }
]

df_input = pd.DataFrame(raw_data)

# ==========================================
# 3. THE CRASH-PROOF EXPLOSION
# ==========================================

# STEP A: Explode Line Items (J_Codes & Units)
# CRITICAL FIX: We use .reset_index(drop=True) to fix the duplicate axis error.
# We assume J_Codes and Units are paired 1-to-1.
df_lines = df_input.explode(['J_Codes', 'Units'])
df_lines = df_lines.reset_index(drop=True) # <--- THIS FIXES YOUR ERROR

# STEP B: Explode Context (Diagnoses)
# Now we cross-join the diagnoses.
df_pairs = df_lines.explode('Diagnoses')
df_pairs = df_pairs.reset_index(drop=True) # <--- THIS FIXES YOUR ERROR AGAIN

# Ensure types
df_pairs = df_pairs.rename(columns={'Diagnoses': 'Raw_Diagnosis', 'J_Codes': 'J_Code'})
df_pairs['Units'] = pd.to_numeric(df_pairs['Units'])

# ==========================================
# 4. PREDICT & AGGREGATE
# ==========================================

# Preprocess
df_pairs = clean_diagnosis(df_pairs, 'Raw_Diagnosis')
df_pairs['Diag_Encoded'] = le_diag.transform(df_pairs['Diag_Category'])
df_pairs['JCode_Encoded'] = le_jcode.transform(df_pairs['J_Code'])

# Predict
df_pairs['Anomaly_Score'] = iso_forest.decision_function(df_pairs[features])

# Aggregation Logic
final_report = df_pairs.groupby(['Claim_ID', 'J_Code', 'Units']).agg(
    Best_Score=('Anomaly_Score', 'max')
).reset_index()

# Reason Generation
def get_status(row):
    if row['Best_Score'] > 0:
        return "APPROVED"
    if row['Units'] > 100:
        return f"FLAG: Excessive Units ({row['Units']})"
    return "FLAG: Clinical Mismatch"

final_report['Status'] = final_report.apply(get_status, axis=1)

print("\n--- FINAL ROBUST REPORT ---")
print(final_report.to_string(index=False))
