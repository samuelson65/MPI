import pandas as pd
import numpy as np
from pyspark.sql import SparkSession

# Initialize Spark Session (Standard for Databricks)
spark = SparkSession.builder.appName("ClaimsAuditProduction").getOrCreate()

# ==========================================
# 1. DATA INGESTION (Simulated for Production)
# ==========================================

# TABLE A: RAW CLAIMS DATA
# Includes duplicate claim numbers to test deduplication logic
claims_data = {
    'claimnumber': [f'CLM{i}' for i in range(1001, 1021)] + ['CLM1001', 'CLM1002'], # Added Duplicates
    'providertaxid': ['TAX_999', 'TAX_111', 'TAX_999', 'TAX_222', 'TAX_111', 'TAX_333', 'TAX_999', 'TAX_222', 'TAX_111', 'TAX_444', 'TAX_555', 'TAX_111', 'TAX_999', 'TAX_111', 'TAX_222', 'TAX_999', 'TAX_999', 'TAX_111', 'TAX_555', 'TAX_999', 'TAX_111'],
    'hcpccode': [
        'J9271', 'J9271', 'J9271', 'J9271',  # Keytruda
        'J0896', 'J0896', 'J0896', 'J0896',  # Reblozyl
        '85025', '85025', '85025',           # Lab
        '99213', '99213', '99213', '99213',  # Office Visit
        'J9999', 'J9999', 'J3490', 'J3590',   # Unlisted
        'J9271', 'J9271' # Duplicates
    ],
    # Messy formats: strings with $, commas, NaNs
    'paidamount': ['$5,000.00', '500', 5000, 6000, 1500, 2000, 1500, 2000, 15, 15, 50, 150, 100, 150, 100, 3000, 3200, 400, 150, '$5,000.00', '500'],
    'unitcount': ['10', 1, 10, 10, 50, 50, 50, 50, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, '10', 1],
    'date_of_service': pd.date_range(start='1/1/2024', periods=19).tolist() + ['2024-01-01', '2024-01-02']
}

# Ensure list lengths align
max_len = max(len(v) for v in claims_data.values())
for k in claims_data:
    if len(claims_data[k]) < max_len: claims_data[k] = claims_data[k] * (max_len // len(claims_data[k]) + 1)
    claims_data[k] = claims_data[k][:max_len]

df_claims_raw = pd.DataFrame(claims_data)

# TABLE B: REFERENCE DATA (The Source of Truth)
ref_data = {
    'hcpccode': ['J9271', 'J0896', '85025', '99213', 'L0650', 'J9999', 'J3490', 'J3590'],
    'description': ['Keytruda 1mg', 'Reblozyl 0.25mg', 'Complete Blood Count', 'Office Visit Level 3', 'Lumbar Orthosis', 'Unlisted Chemo', 'Unlisted Bio', 'Unlisted'],
    'category': ['Oncology', 'Specialty', 'Lab', 'E&M', 'DME', 'Unlisted', 'Unlisted', 'Unlisted'],
    'min_allowed': [480, 30, 10, 90, 100, 0, 0, 0],
    'max_allowed': [520, 50, 20, 120, 200, 0, 0, 0]
}
df_ref = pd.DataFrame(ref_data)

# ==========================================
# 2. THE PRODUCTION PIPELINE (Clean -> Dedup -> Join)
# ==========================================

def run_production_pipeline(df_claims, df_ref):
    print("🚀 STARTING PIPELINE...")
    
    # --- STEP 1: CLEANING (The "Bronze" Layer) ---
    # Handle currency strings, commas, and mixed types
    clean_cols = ['paidamount', 'unitcount']
    for col in clean_cols:
        df_claims[col] = df_claims[col].astype(str).str.replace(r'[$,]', '', regex=True)
        df_claims[col] = pd.to_numeric(df_claims[col], errors='coerce').fillna(0)
    
    # Fix Dates
    df_claims['date_of_service'] = pd.to_datetime(df_claims['date_of_service'], errors='coerce')
    
    # --- STEP 2: DEDUPLICATION (The Critical Fix) ---
    # We define a "Duplicate" as same Claim Number + Same Code + Same Date
    initial_count = len(df_claims)
    df_claims = df_claims.drop_duplicates(subset=['claimnumber', 'hcpccode', 'date_of_service'], keep='first')
    dedup_count = len(df_claims)
    
    if initial_count > dedup_count:
        print(f"⚠️  DEDUPLICATION ALERT: Removed {initial_count - dedup_count} duplicate rows.")
    else:
        print("✅  Data Integrity Check Passed: No duplicates found.")

    # --- STEP 3: METRIC CALCULATION ---
    # Calculate Unit Cost (Handle division by zero safety)
    df_claims['unit_cost'] = df_claims['paidamount'] / df_claims['unitcount'].replace(0, 1)

    # --- STEP 4: ENRICHMENT (Join with Ref) ---
    # Left Join: Keep all claims, even if missing from Ref Table (to catch "Unknown Codes")
    df_merged = df_claims.merge(df_ref, on='hcpccode', how='left')
    
    # Fill NaN categories for reporting
    df_merged['category'] = df_merged['category'].fillna('Unknown/Missing Ref')
    df_merged['description'] = df_merged['description'].fillna('Unknown Desc')
    df_merged['max_allowed'] = df_merged['max_allowed'].fillna(0) # Default to 0 limit if unknown

    # --- STEP 5: LOGIC FLAGS ---
    # Flag Overpayments (Allowing for $0 max_allowed as a pass if category is Unlisted)
    # Logic: If max_allowed > 0 AND unit_cost > max_allowed -> Overpayment
    df_merged['is_overpaid'] = (df_merged['unit_cost'] > df_merged['max_allowed']) & (df_merged['max_allowed'] > 0)
    
    df_merged['overpayment_amt'] = np.where(
        df_merged['is_overpaid'], 
        df_merged['paidamount'] - (df_merged['max_allowed'] * df_merged['unitcount']), 
        0
    )
    
    print("✅  PIPELINE COMPLETE.")
    return df_merged

# Run the Pipeline
df_audit = run_production_pipeline(df_claims_raw, df_ref)

# ==========================================
# 3. GENERATE STRATEGIC OUTPUTS (Display)
# ==========================================

print("\n---------------------------------------------------------")
print("📥 VIEW 1: STRATEGIC SPEND PARETO (Category Level)")
print("   ACTION: Download for Executive Summary")
print("---------------------------------------------------------")
view1 = df_audit.groupby(['category', 'hcpccode', 'description']).agg(
    Total_Spend=('paidamount', 'sum'),
    Claim_Volume=('claimnumber', 'count'),
    Avg_Unit_Cost=('unit_cost', 'mean')
).sort_values(['category', 'Total_Spend'], ascending=[True, False]).reset_index()
display(view1)


print("\n---------------------------------------------------------")
print("📥 VIEW 2: FEE SCHEDULE RECOVERY LIST (Audit Targets)")
print("   ACTION: Send to Recovery Team. These are contract violations.")
print("---------------------------------------------------------")
view2 = df_audit[df_audit['is_overpaid']].copy()
view2 = view2[['claimnumber', 'providertaxid', 'hcpccode', 'description', 'unit_cost', 'max_allowed', 'overpayment_amt']]
view2 = view2.sort_values('overpayment_amt', ascending=False)
display(view2)


print("\n---------------------------------------------------------")
print("📥 VIEW 3: PROVIDER VARIANCE SCORECARD")
print("   ACTION: Identify providers billing >20% above peers.")
print("---------------------------------------------------------")
# 1. Calc Provider Average
prov_stats = df_audit.groupby(['category', 'providertaxid']).agg(
    Prov_Avg_Cost=('unit_cost', 'mean'),
    Prov_Spend=('paidamount', 'sum'),
    Prov_Claims=('claimnumber', 'count')
).reset_index()

# 2. Calc Category Benchmark (The "Norm")
cat_benchmark = df_audit.groupby('category')['unit_cost'].mean().reset_index().rename(columns={'unit_cost': 'Benchmark_Cost'})

# 3. Join & Score
view3 = prov_stats.merge(cat_benchmark, on='category', how='left')
view3['Variance_%'] = ((view3['Prov_Avg_Cost'] - view3['Benchmark_Cost']) / view3['Benchmark_Cost']) * 100

# Filter for meaningful variance (e.g., > 10%)
view3 = view3[view3['Variance_%'] > 10].sort_values('Variance_%', ascending=False)
display(view3)


print("\n---------------------------------------------------------")
print("📥 VIEW 4: UNKNOWN / UNLISTED RISK LOG")
print("   ACTION: These codes had NO reference data or were Unlisted.")
print("---------------------------------------------------------")
view4 = df_audit[df_audit['category'].isin(['Unknown/Missing Ref', 'Unlisted'])].copy()
view4 = view4.groupby(['providertaxid', 'hcpccode']).agg(
    Total_Risky_Spend=('paidamount', 'sum'),
    Claims=('claimnumber', 'count')
).sort_values('Total_Risky_Spend', ascending=False).reset_index()
display(view4)
