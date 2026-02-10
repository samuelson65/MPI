import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. GENERATE REALISTIC RAW DATA
# ==========================================
# Notice: We use FULL ICD-10 codes here (e.g., E11.9, J45.909)
# The model will automatically "learn" that these belong to the same parent.

np.random.seed(42)

# Group A: Diabetes Variations (E11.9, E11.65, E11.2) -> Insulin
df_diabetes = pd.DataFrame({
    'Raw_Diagnosis': np.random.choice(['E11.9', 'E11.65', 'E11.2', 'E11.4'], 4000),
    'J_Code': ['J1815'] * 4000,
    'Units': np.random.normal(30, 5, 4000)
})

# Group B: Asthma Variations (J45.909, J45.901) -> Albuterol
df_asthma = pd.DataFrame({
    'Raw_Diagnosis': np.random.choice(['J45.909', 'J45.901', 'J45.4'], 4000),
    'J_Code': ['J7613'] * 4000,
    'Units': np.random.normal(10, 2, 4000)
})

# Group C: Anomalies (Random Noise for training)
df_noise = pd.DataFrame({
    'Raw_Diagnosis': np.random.choice(['E11.9', 'J45.909', 'S82.1'], 500),
    'J_Code': np.random.choice(['J9999', 'J3490'], 500),
    'Units': np.random.uniform(50, 100, 500)
})

df_train = pd.concat([df_diabetes, df_asthma, df_noise]).sample(frac=1).reset_index(drop=True)

# ==========================================
# 2. SMART PRE-PROCESSING (The 3-Digit Logic)
# ==========================================

def preprocess_diagnosis(df):
    # logic: Convert to string -> Slice first 3 chars -> Uppercase
    df['Diag_Category'] = df['Raw_Diagnosis'].astype(str).str[:3].str.upper()
    return df

# Apply the logic
df_train = preprocess_diagnosis(df_train)

# ==========================================
# 3. ENCODING & TRAINING
# ==========================================
le_diag = LabelEncoder()
le_jcode = LabelEncoder()

# Fit the encoders on the GENERALIZED 3-digit codes
df_train['Diag_Encoded'] = le_diag.fit_transform(df_train['Diag_Category'])
df_train['JCode_Encoded'] = le_jcode.fit_transform(df_train['J_Code'])

# Features: [Generalized Diag, Exact Drug, Quantity]
features = ['Diag_Encoded', 'JCode_Encoded', 'Units']

# Train the "Brain"
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(df_train[features])

print(f"--- Model Trained on {len(df_train)} Claims ---")
print(f"Learned Categories: {le_diag.classes_}") # Should see 'E11', 'J45', 'S82'
print("-" * 30)

# ==========================================
# 4. PREDICT NEW (UNSEEN) CLAIMS
# ==========================================
# Let's test with a NEW, highly specific diagnosis code the model has NEVER seen.
# e.g., 'E11.319' (Diabetes w/ Retinopathy) - The model only knows 'E11'
new_claims = [
    # Case 1: New Specific Code (E11.319) + Correct Drug
    # Result: SHOULD APPROVE (Because E11.319 becomes E11)
    {'Raw_Diagnosis': 'E11.319', 'J_Code': 'J1815', 'Units': 30},
    
    # Case 2: New Specific Code (J45.2) + Wrong Drug (Insulin)
    # Result: SHOULD FLAG (Because J45 + Insulin is rare)
    {'Raw_Diagnosis': 'J45.2', 'J_Code': 'J1815', 'Units': 10},
    
    # Case 3: Totally Random Code (R51 Headache) + Chemo
    # Result: SHOULD FLAG (New Category + New Pattern)
    {'Raw_Diagnosis': 'R51', 'J_Code': 'J9000', 'Units': 5}
]

df_new = pd.DataFrame(new_claims)

# 1. Apply same 3-digit logic
df_new = preprocess_diagnosis(df_new)

# 2. Encode (Handle unknown categories safely)
# If a category is brand new (e.g., R51), we assign a special "Unknown" code or -1
# For this demo, we use a helper to handle unseen labels
def safe_transform(encoder, series):
    return series.map(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

df_new['Diag_Encoded'] = safe_transform(le_diag, df_new['Diag_Category'])
df_new['JCode_Encoded'] = safe_transform(le_jcode, df_new['J_Code'])

# 3. Predict
df_new['Anomaly_Score'] = iso_forest.decision_function(df_new[features])
df_new['Prediction'] = iso_forest.predict(df_new[features])

# ==========================================
# 5. GENERATE SMART OUTPUT
# ==========================================
df_new['Status'] = np.where(df_new['Prediction'] == 1, 'APPROVED', 'FLAGGED')

print(df_new[['Raw_Diagnosis', 'Diag_Category', 'J_Code', 'Units', 'Status', 'Anomaly_Score']])
