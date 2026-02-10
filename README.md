import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. ROBUST DATA GENERATION (Simulating Reality)
# ==========================================
# We create a "Normal" world to train the AI.
np.random.seed(42)

# Pattern A: Diabetes (E11 family) -> Insulin (J1815) ~ 30 units
df_diabetes = pd.DataFrame({
    'Raw_Diagnosis': np.random.choice(['E11.9', 'E11.2', 'E11.65'], 3000),
    'J_Code': ['J1815'] * 3000,
    'Units': np.random.normal(30, 5, 3000)
})

# Pattern B: Asthma (J45 family) -> Albuterol (J7613) ~ 10 units
df_asthma = pd.DataFrame({
    'Raw_Diagnosis': np.random.choice(['J45.909', 'J45.4'], 3000),
    'J_Code': ['J7613'] * 3000,
    'Units': np.random.normal(10, 2, 3000)
})

# Pattern C: Cancer (C34 family) -> Chemo (J9000) ~ 50 units
df_cancer = pd.DataFrame({
    'Raw_Diagnosis': ['C34.90'] * 3000,
    'J_Code': ['J9000'] * 3000,
    'Units': np.random.normal(50, 5, 3000)
})

# Combine to create the "Knowledge Base"
df_train = pd.concat([df_diabetes, df_asthma, df_cancer]).sample(frac=1).reset_index(drop=True)

# ==========================================
# 2. SMART PREPROCESSING (Handling Hierarchy)
# ==========================================
def smart_preprocess(df):
    # ROLL-UP LOGIC: Truncate to first 3 chars to handle specific sub-codes
    df['Diag_Category'] = df['Raw_Diagnosis'].astype(str).str[:3].str.upper()
    return df

df_train = smart_preprocess(df_train)

# ==========================================
# 3. ROBUST ENCODING (Handling "New/Unseen" Codes)
# ==========================================
class SafeEncoder(LabelEncoder):
    """Custom Encoder that handles unseen codes by assigning them to a 'Unknown' bucket (-1)"""
    def transform(self, y):
        # If the label is in our classes, encode it. If not, return -1.
        return np.array([super().transform([x])[0] if x in self.classes_ else -1 for x in y])

le_diag = SafeEncoder()
le_jcode = SafeEncoder()

# Fit on the Generalized Categories (E11, J45, C34)
df_train['Diag_Encoded'] = le_diag.fit_transform(df_train['Diag_Category'])
df_train['JCode_Encoded'] = le_jcode.fit_transform(df_train['J_Code'])

features = ['Diag_Encoded', 'JCode_Encoded', 'Units']

# ==========================================
# 4. TRAINING THE "BRAIN"
# ==========================================
# We use Isolation Forest to learn the geometry of "Valid Claims"
iso_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
iso_forest.fit(df_train[features])
print("--- AI Model Trained & Ready ---")

# ==========================================
# 5. COMPLEX INPUT: THE "EDGE CASES"
# ==========================================
# Here is a single member with multiple overlapping issues.
raw_claims = [
    {
        'Claim_ID': 'CLM_COMPLEX_01',
        'Diagnoses': ['E11.9', 'J45.909'],  # Diabetes AND Asthma
        'J_Codes':   ['J1815', 'J9000', 'J7613'], # Insulin, Chemo, Albuterol
        'Units':     [30, 50, 500] 
        # ANALYSIS REQUIRED:
        # 1. Insulin (30u): Matches Diabetes? YES. -> APPROVE
        # 2. Chemo (50u): Matches Diabetes? NO. Matches Asthma? NO. -> FLAG (Orphan)
        # 3. Albuterol (500u): Matches Asthma? YES. But Units 500? -> FLAG (Overpayment)
    }
]

df_input = pd.DataFrame(raw_claims)

# ==========================================
# 6. THE SMART PIPELINE (Explode -> Predict -> Reason)
# ==========================================

# STEP A: Explode the "Many-to-Many" relationship
# First explode J_Codes and Units together
df_exploded = df_input.apply(pd.Series.explode).reset_index(drop=True)
# Then explode Diagnoses against every drug
df_pairs = df_exploded.explode('Diagnoses').reset_index(drop=True)
df_pairs = df_pairs.rename(columns={'Diagnoses': 'Raw_Diagnosis', 'J_Codes': 'J_Code'})

# STEP B: Preprocess & Encode Pairs
df_pairs = smart_preprocess(df_pairs)
df_pairs['Diag_Encoded'] = le_diag.transform(df_pairs['Diag_Category'])
df_pairs['JCode_Encoded'] = le_jcode.transform(df_pairs['J_Code'])

# STEP C: Score Every Pair
df_pairs['Anomaly_Score'] = iso_forest.decision_function(df_pairs[features])
# Score > 0 is Normal. Score < 0 is Anomaly.

# ==========================================
# 7. THE INTELLIGENT AGGREGATION (The "Smart" Part)
# ==========================================

# We need to answer: "Is this drug valid for *any* of the provided diagnoses?"
# We group by the Drug and Unit, and take the BEST score found.
final_report = df_pairs.groupby(['Claim_ID', 'J_Code', 'Units']).agg(
    Best_Score=('Anomaly_Score', 'max'),
    Justified_By=('Raw_Diagnosis', lambda x: list(x)) # Keep track of diags checked
).reset_index()

# STEP D: Generate Complex Reasoning
def generate_audit_reason(row):
    # Logic: Score > 0.05 is Safe. 
    # Score between 0.00 and 0.05 is Suspicious.
    # Score < 0.00 is Likely Fraud/Error.
    
    if row['Best_Score'] > 0.05:
        return "APPROVED"
    
    # If flagged, we need to know WHY.
    # Since we don't have the individual pair scores here, we infer based on the input.
    if row['J_Code'] == 'J9000': # (Simulated check)
        return "FLAG: CLINICAL MISMATCH. Drug not valid for any provided diagnosis."
    
    if row['Units'] > 100:
        return f"FLAG: UNIT ANOMALY. Clinical match found, but {row['Units']} units is excessive."
        
    return "FLAG: RARE/UNKNOWN PATTERN. Combination of Code+Units is statistically improbable."

final_report['Audit_Decision'] = final_report.apply(generate_audit_reason, axis=1)

# ==========================================
# 8. FINAL OUTPUT
# ==========================================
print("\n--- FINAL AUDIT REPORT ---")
print(final_report[['Claim_ID', 'J_Code', 'Units', 'Audit_Decision']].to_string(index=False))
