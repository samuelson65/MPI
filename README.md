import pandas as pd
import numpy as np

# ==========================================
# 1. SMART TRAINING (The "Brain")
# ==========================================
# We simulate a large historical dataset of 10,000+ valid claims.
# The engine uses this to learn "Clinical Categories" automatically.

# Mocking the historical data
data = {
    'Diagnosis_Code': (
        ['E11'] * 5000 +    # 5000 cases of Diabetes (E11)
        ['J45'] * 3000 +    # 3000 cases of Asthma (J45)
        ['S82'] * 2000      # 2000 cases of Fractures (S82)
    ),
    'J_Code': (
        ['J_INSULIN'] * 4500 + ['J_METFORMIN'] * 500 +  # Diabetes drugs
        ['J_ALBUTEROL'] * 2800 + ['J_STEROID'] * 200 +  # Asthma drugs
        ['J_CAST'] * 1900 + ['J_SPLINT'] * 100          # Fracture procedures
    )
}
df_history = pd.DataFrame(data)

# TRAIN THE MODEL: Calculate the "Affinity Score" between every Diag and J-Code
# This tells us: "How clinically similar is this drug to this disease?"
model_affinity = df_history.groupby(['Diagnosis_Code', 'J_Code']).size().reset_index(name='Count')
total_per_diag = df_history.groupby('Diagnosis_Code').size().reset_index(name='Total')
model_affinity = model_affinity.merge(total_per_diag, on='Diagnosis_Code')
model_affinity['Score_Pct'] = (model_affinity['Count'] / model_affinity['Total']) * 100

# BUILD THE "CLINICAL COMPARATOR" (What SHOULD they have used?)
# For every Diagnosis, memorize the #1 most common treatment.
idx = model_affinity.groupby(['Diagnosis_Code'])['Count'].transform(max) == model_affinity['Count']
clinical_gold_standard = model_affinity[idx][['Diagnosis_Code', 'J_Code']].rename(
    columns={'J_Code': 'Recommended_Treatment'}
)

print("--- Engine Trained on History (Sample Rules) ---")
print(model_affinity.head(3))
print("-" * 30)

# ==========================================
# 2. THE INPUT (Complex Claims)
# ==========================================
# New claims come in with LISTS of Diagnoses and LISTS of J-Codes.
raw_new_claims = [
    {
        'Claim_ID': 101,
        'Member_ID': 'MEM_A',
        'Diagnoses': ['E11', 'S82'], # Diabetes + Fracture
        'J_Codes':   ['J_INSULIN', 'J_CAST'] # Both Valid (Cross-match)
    },
    {
        'Claim_ID': 102,
        'Member_ID': 'MEM_B',
        'Diagnoses': ['J45'], # Asthma
        'J_Codes':   ['J_INSULIN'] # INVALID (Diabetes drug for Asthma?)
    },
    {
        'Claim_ID': 103,
        'Member_ID': 'MEM_C',
        'Diagnoses': ['S82', 'J45'], # Fracture + Asthma
        'J_Codes':   ['J_ALBUTEROL', 'J_UNKNOWN_DRUG'] # 1 Valid, 1 Invalid
    }
]
df_new = pd.DataFrame(raw_new_claims)

# ==========================================
# 3. EFFICIENT VECTORIZED PROCESSING
# ==========================================

# STEP A: Explode J_Codes (Analyze each drug individually)
df_exploded_j = df_new.explode('J_Codes')

# STEP B: Explode Diagnoses (Create all possible Diag-Drug pairs)
# If a member has 2 Diags and 2 Drugs, this creates 4 rows to check.
df_pairs = df_exploded_j.explode('Diagnoses')

# STEP C: Vectorized Lookup (The Fast Match)
# Merge with the trained model to get scores for every possible pair
df_scored = df_pairs.merge(
    model_affinity[['Diagnosis_Code', 'J_Code', 'Score_Pct']], 
    left_on=['Diagnoses', 'J_Codes'], 
    right_on=['Diagnosis_Code', 'J_Code'], 
    how='left'
)

# Fill unknown pairs with 0 score
df_scored['Score_Pct'] = df_scored['Score_Pct'].fillna(0)

# STEP D: Aggregation (The "Any-Match" Logic)
# For each specific J-Code on a claim, take the BEST score found across all diagnoses.
# If "Insulin" matches "Diabetes" (100%) but not "Fracture" (0%), the Max is 100% (Valid).
final_decisions = df_scored.groupby(['Claim_ID', 'J_Codes'])['Score_Pct'].max().reset_index()

# ==========================================
# 4. CLINICAL COMPARISON & REASONING
# ==========================================

def get_comparison(row):
    # If the score is high, no comparison needed.
    if row['Score_Pct'] > 5.0: 
        return "MATCH: Clinically Appropriate"
    
    # If score is low, find the specific diagnosis that caused the mismatch
    # We look back at the input claim to find the likely intended diagnosis
    # (Simplified: We just grab the Rec from the first diagnosis on the list for this example)
    return "MISMATCH"

final_decisions['Status'] = np.where(final_decisions['Score_Pct'] > 5.0, 'APPROVED', 'FLAGGED')

# MERGE BACK "SMART COMPARISON"
# We join the "Gold Standard" table to show what SHOULD have been used.
# Since we lost the 'Diagnoses' column during groupby, we grab the first diagnosis from original data for context
df_context = df_new[['Claim_ID', 'Diagnoses']].explode('Diagnoses').merge(
    clinical_gold_standard, left_on='Diagnoses', right_on='Diagnosis_Code', how='left'
)
# Aggregate likely treatments into a string
df_context_grouped = df_context.groupby('Claim_ID')['Recommended_Treatment'].apply(lambda x: ', '.join(set(x))).reset_index()

# Final Join
final_report = final_decisions.merge(df_context_grouped, on='Claim_ID')

# Logic for the "Smart Reason" Column
def construct_smart_reason(row):
    if row['Status'] == 'APPROVED':
        return f"Approved. Validated by history."
    else:
        return f"Outlier (Score {row['Score_Pct']:.1f}). Clinical Mismatch. Patients with these diagnoses typically use: [{row['Recommended_Treatment']}]"

final_report['Analysis'] = final_report.apply(construct_smart_reason, axis=1)

# ==========================================
# 5. OUTPUT
# ==========================================
print("\n--- FINAL SMART REPORT ---")
print(final_report[['Claim_ID', 'J_Codes', 'Status', 'Analysis']].to_string(index=False))
