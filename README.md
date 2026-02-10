import pandas as pd
import numpy as np

# ==========================================
# 1. SMART TRAINING (With Hierarchy)
# ==========================================
# We train TWO models: one for specific codes, one for broad categories.

# Mock History:
# - E11 (General Diabetes) uses Insulin
# - J45.909 (Specific Asthma) uses Albuterol
history_data = {
    'Diagnosis_Code': (
        ['E11'] * 1000 +        # General code used in history
        ['J45.909'] * 1000      # Specific code used in history
    ),
    'J_Code': (
        ['J_INSULIN'] * 1000 +  # Valid for E11
        ['J_ALBUTEROL'] * 1000  # Valid for J45.909
    )
}
df_history = pd.DataFrame(history_data)

# Create Parent Column (First 3 chars)
df_history['Parent_Code'] = df_history['Diagnosis_Code'].str[:3]

# --- MODEL A: EXACT MATCH (High Precision) ---
model_exact = df_history.groupby(['Diagnosis_Code', 'J_Code']).size().reset_index(name='Count')
total_exact = df_history.groupby('Diagnosis_Code').size().reset_index(name='Total')
model_exact = model_exact.merge(total_exact, on='Diagnosis_Code')
model_exact['Exact_Score'] = (model_exact['Count'] / model_exact['Total']) * 100

# --- MODEL B: PARENT ROLL-UP (Broad Coverage) ---
# Group by Parent Code (e.g., 'E11') instead of specific code
model_parent = df_history.groupby(['Parent_Code', 'J_Code']).size().reset_index(name='Count')
total_parent = df_history.groupby('Parent_Code').size().reset_index(name='Total')
model_parent = model_parent.merge(total_parent, on='Parent_Code')
model_parent['Parent_Score'] = (model_parent['Count'] / model_parent['Total']) * 100

print("--- Trained Models ---")
print(f"Exact Rules: {len(model_exact)} | Parent Rules: {len(model_parent)}")
print("-" * 30)

# ==========================================
# 2. THE INPUT (New Claims)
# ==========================================
raw_new_claims = [
    {
        'Claim_ID': 101,
        'Member_ID': 'MEM_Specific',
        'Diagnoses': ['E11.9'],  # Specific Code (Not in history, but E11 is)
        'J_Codes':   ['J_INSULIN'] # Should Pass via Roll-Up
    },
    {
        'Claim_ID': 102,
        'Member_ID': 'MEM_Mismatch',
        'Diagnoses': ['J45'],      # General Code
        'J_Codes':   ['J_INSULIN'] # Invalid (Asthma != Insulin)
    }
]
df_new = pd.DataFrame(raw_new_claims)

# ==========================================
# 3. PROCESSING WITH ROLL-UP LOOKUP
# ==========================================

# Explode lists to rows
df_exploded = df_new.explode('J_Codes').explode('Diagnoses')

# Create Parent Column for New Claims
df_exploded['Parent_Code'] = df_exploded['Diagnoses'].str[:3]

# STEP A: Try Exact Match First
df_scored = df_exploded.merge(
    model_exact[['Diagnosis_Code', 'J_Code', 'Exact_Score']], 
    left_on=['Diagnoses', 'J_Codes'], 
    right_on=['Diagnosis_Code', 'J_Code'], 
    how='left'
)

# STEP B: Try Parent Match Second (The Roll-Up)
df_scored = df_scored.merge(
    model_parent[['Parent_Code', 'J_Code', 'Parent_Score']], 
    left_on=['Parent_Code', 'J_Code'], 
    right_on=['Parent_Code', 'J_Code'], 
    how='left'
)

# Fill NaNs with 0
df_scored['Exact_Score'] = df_scored['Exact_Score'].fillna(0)
df_scored['Parent_Score'] = df_scored['Parent_Score'].fillna(0)

# STEP C: The Decision Logic (Prioritize Exact, Fallback to Parent)
def determine_final_status(row):
    # 1. If Exact Match exists and is good -> APPROVED
    if row['Exact_Score'] > 5.0:
        return "APPROVED (Exact Match)"
    
    # 2. If Exact Match failed, check Parent -> APPROVED
    if row['Parent_Score'] > 5.0:
        return "APPROVED (Roll-Up Match)"
    
    # 3. If both fail -> FLAG
    return "FLAGGED (Mismatch)"

df_scored['Status'] = df_scored.apply(determine_final_status, axis=1)

# ==========================================
# 4. FINAL REPORT
# ==========================================
final_output = df_scored[['Claim_ID', 'Diagnoses', 'J_Codes', 'Exact_Score', 'Parent_Score', 'Status']]

print("\n--- FINAL ROLL-UP REPORT ---")
print(final_output.to_string(index=False))
