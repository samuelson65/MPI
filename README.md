# ... (Assume previous setup code is here) ...

# ==========================================
# ENHANCEMENT: TRAIN FOR UNITS (QUANTITY)
# ==========================================
# We calculate the "Normal Limit" for every valid pair
unit_stats = df_history.groupby(['Diagnosis_Code', 'J_Code'])['Units'].agg(
    Avg_Units='mean',
    Std_Dev='std'
).reset_index()

# Fill NaN std dev (for single occurrences) with 0
unit_stats['Std_Dev'] = unit_stats['Std_Dev'].fillna(0)

# Define the "Safe Limit" (Average + 3 Standard Deviations)
# If Std_Dev is 0 (consistent billing), we allow a small buffer (e.g., 20% more)
unit_stats['Max_Safe_Units'] = np.where(
    unit_stats['Std_Dev'] == 0, 
    unit_stats['Avg_Units'] * 1.2, 
    unit_stats['Avg_Units'] + (3 * unit_stats['Std_Dev'])
)

# ==========================================
# APPLYING IT TO NEW CLAIMS
# ==========================================
# 1. Merge the Unit Stats into your scored dataframe
df_scored_units = df_scored.merge(
    unit_stats[['Diagnosis_Code', 'J_Code', 'Max_Safe_Units']],
    on=['Diagnosis_Code', 'J_Code'],
    how='left'
)

# 2. Check for Overpayment (Excessive Units)
# Logic: If Drug is Valid (High Score) BUT Units are too high -> FLAG OVERPAYMENT
def check_overpayment(row):
    if row['Score_Pct'] < 5.0:
        return "Mismatch: Drug not valid for Diagnosis"
    
    # If units are missing in claim, assume 1 (or handle error)
    claimed_units = 100 # Example: Hardcoded for demo, normally comes from row['Units']
    
    if claimed_units > row['Max_Safe_Units']:
        return f"OVERPAYMENT RISK: {claimed_units} units billed. Norm is {row['Max_Safe_Units']:.1f}."
    
    return "APPROVED"

# Apply the logic
df_scored_units['Financial_Review'] = df_scored_units.apply(check_overpayment, axis=1)

print(df_scored_units[['Diagnosis_Code', 'J_Code', 'Financial_Review']].head())
