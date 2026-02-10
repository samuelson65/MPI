import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. SETUP TRAIN DATA (No changes here)
# ==========================================
np.random.seed(42)

# Training Data
df_train = pd.DataFrame({
    'Raw_Diagnosis': (['E11.9'] * 2000 + ['J45.909'] * 2000 + ['C34.90'] * 2000),
    'J_Code':        (['J1815'] * 2000 + ['J7613'] * 2000 + ['J9000'] * 2000),
    'Units':         (np.random.normal(30, 2, 2000).tolist() + 
                      np.random.normal(10, 2, 2000).tolist() + 
                      np.random.normal(50, 5, 2000).tolist())
})

# Preprocessing
def smart_preprocess(df):
    df['Diag_Category'] = df['Raw_Diagnosis'].astype(str).str[:3].str.upper()
    return df

df_train = smart_preprocess(df_train)

# Encoding
class SafeEncoder(LabelEncoder):
    def transform(self, y):
        # Handle unseen codes safely
        return np.array([super().transform([x])[0] if x in self.classes_ else -1 for x in y])

le_diag = SafeEncoder()
le_jcode = SafeEncoder()

df_train['Diag_Encoded'] = le_diag.fit_transform(df_train['Diag_Category'])
df_train['JCode_Encoded'] = le_jcode.fit_transform(df_train['J_Code'])

features = ['Diag_Encoded', 'JCode_Encoded', 'Units']

# Train Model
iso_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
iso_forest.fit(df_train[features])
print("--- AI Model Trained ---")

# ==========================================
# 2. COMPLEX INPUT (The Fix is below)
# ==========================================
raw_claims = [
    {
        'Claim_ID': 'CLM_ERROR_FIXED',
        'Diagnoses': ['E11.9', 'J45.909'],  # Length 2
        'J_Codes':   ['J1815', 'J9000', 'J7613'], # Length 3
        'Units':     [30, 50, 500]        # Length 3 (Matches J_Codes)
    }
]

df_input = pd.DataFrame(raw_claims)

# ==========================================
# 3. THE FIXED EXPLOSION LOGIC
# ==========================================

# FIX: Do NOT explode everything at once.
# Step A: Explode the "Line Items" (J_Codes and Units must match)
# We leave 'Diagnoses' as a list for a moment.
df_line_items = df_input.explode(['J_Codes', 'Units']).reset_index(drop=True)

# Step B: Explode the "Context" (Diagnoses)
# Now we explode the diagnoses so that every Drug Line gets paired with every Diagnosis.
df_pairs = df_line_items.explode('Diagnoses').reset_index(drop=True)

# Rename for clarity
df_pairs = df_pairs.rename(columns={'Diagnoses': 'Raw_Diagnosis', 'J_Codes': 'J_Code'})

# ==========================================
# 4. PREDICT & AGGREGATE
# ==========================================

# Preprocess & Encode
df_pairs = smart_preprocess(df_pairs)
df_pairs['Diag_Encoded'] = le_diag.transform(df_pairs['Diag_Category'])
df_pairs['JCode_Encoded'] = le_jcode.transform(df_pairs['J_Code'])

# Predict
df_pairs['Anomaly_Score'] = iso_forest.decision_function(df_pairs[features])

# Aggregation Logic (Find best justifying diagnosis)
final_report = df_pairs.groupby(['Claim_ID', 'J_Code', 'Units']).agg(
    Best_Score=('Anomaly_Score', 'max')
).reset_index()

# Generate Reason
def generate_audit_reason(row):
    # Score > 0 is Normal (Inliers)
    # Score < 0 is Anomaly (Outliers)
    if row['Best_Score'] > 0:
        return "APPROVED"
    else:
        # Simple heuristics for the reason
        if row['Units'] > 100:
            return f"FLAG: Excessive Units ({row['Units']})"
        return "FLAG: Clinical Mismatch"

final_report['Audit_Decision'] = final_report.apply(generate_audit_reason, axis=1)

print("\n--- FINAL FIXED REPORT ---")
print(final_report.to_string(index=False))
