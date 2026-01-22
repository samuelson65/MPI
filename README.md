import pandas as pd
import numpy as np

# ==========================================
# 1. SAMPLE DATA (Simulating "Messy" Real World Data)
# ==========================================
# Note: I've included string formatting errors ('$5,000') to test the cleaning logic
raw_data = {
    'claimnumber': [f'CLM{i}' for i in range(1001, 1016)],
    'hcpccode': [
        'J9271', 'J9271', 'J9271',  # Keytruda (High Variance)
        'J0896', 'J0896', 'J0896',  # Reblozyl (Modifier Impact)
        'L0650', 'L0650',           # DME (Cheap)
        '99213', '99213', '99213', '99213', # Office Visits (Unbundling)
        'J9999', 'J9999', 'J9999'   # Unlisted (High Risk)
    ],
    'mod1': ['None', 'JW', 'None', 'JZ', 'JZ', 'JZ', 'NU', 'NU', '25', 'None', '25', 'None', None, None, 'JW'],
    'mod2': [None, None, None, 'JB', 'JB', 'JB', 'RT', 'LT', None, None, None, None, None, None, None],
    'mod3': [None, None, None, 'TB', None, 'TB', None, None, None, None, None, None, None, None, None],
    'mod4': [None, None, None, 'PO', None, 'PO', None, None, None, None, None, None, None, None, None],
    'mod5': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    # Messy Types: Strings with $, commas, etc.
    'paidamount': ['$5,000.00', '500', 5000, 1500, 2000, '1500', 800, 800, 150, 100, 150, 100, 3000, 3200, 400],
    'chargedamount': ['12000', 1200, 12000, 5000, 6000, 5000, 2500, 2500, 300, 200, 300, 200, 9000, 9500, 1200],
    'date_of_service': [
        '2024-01-01', '01/02/2024', '2024-01-03', '2024-01-05', '2024-01-05', '2024-01-06',
        '2024/02/01', '2024-02-01', '2024-03-01', '2024-03-01', '2024-03-02', '2024-03-03',
        '2024-04-01', '2024-04-02', '2024-04-03'
    ],
    'unitcount': ['10', 1, 10, 50, 50, 50, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

df_raw = pd.DataFrame(raw_data)

# ==========================================
# 2. PRE-PROCESSING (Fix Data Types)
# ==========================================
def clean_and_fix_types(df):
    print("🛠️  STARTING PRE-PROCESSING...")
    
    # 1. Clean Modifiers (Replace NaNs with empty string, uppercase)
    mod_cols = ['mod1', 'mod2', 'mod3', 'mod4', 'mod5']
    for col in mod_cols:
        df[col] = df[col].fillna('').astype(str).str.upper().str.replace('NONE', '')
    
    # Create Signature (Context)
    df['modifier_signature'] = df[mod_cols].apply(lambda x: ','.join(filter(None, x)), axis=1)
    
    # 2. Fix Numeric Columns (Remove '$', ',', convert to float/int)
    numeric_cols = ['paidamount', 'chargedamount', 'unitcount']
    for col in numeric_cols:
        # Convert to string first, strip bad chars, then to numeric
        df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # User Request: "Amount units should all be int" -> Be careful with cents, but rounding to int for view
    # Note: Keeping float for calculations, can display as int later.
    
    # 3. Fix Date Column
    df['date_of_service'] = pd.to_datetime(df['date_of_service'], errors='coerce')
    
    # 4. Calculate Derived Metrics
    # Avoid division by zero by replacing 0 units with 1 temporarily
    df['unit_cost'] = df['paidamount'] / df['unitcount'].replace(0, 1)
    df['markup_ratio'] = df['chargedamount'] / df['paidamount'].replace(0, 1)

    print("✅ TYPES FIXED. Date range:", df['date_of_service'].min().date(), "to", df['date_of_service'].max().date())
    return df

df = clean_and_fix_types(df_raw)

# ==========================================
# 3. INSIGHT GENERATION
# ==========================================

def generate_insights(df):
    print("\n" + "="*50)
    print("📊 DIRECTOR-LEVEL SPEND INSIGHTS")
    print("="*50)

    # --- INSIGHT 1: TOP 5 DRUGS BY SPEND (Pareto Analysis) ---
    top_drugs = df.groupby('hcpccode').agg(
        Total_Spend=('paidamount', 'sum'),
        Claim_Count=('claimnumber', 'count'),
        Avg_Unit_Cost=('unit_cost', 'mean')
    ).sort_values('Total_Spend', ascending=False).head(5)
    
    print("\n1️⃣  TOP SPEND DRIVERS (Pareto)")
    print(top_drugs)

    # --- INSIGHT 2: PRICING VARIANCE (The "Messy Contract" Check) ---
    # High Standard Deviation in Unit Cost means we are paying inconsistent rates
    variance = df.groupby('hcpccode')['unit_cost'].agg(['mean', 'std', 'min', 'max'])
    variance['CV'] = variance['std'] / variance['mean'] # Coefficient of Variation
    high_variance = variance[variance['CV'] > 0.1] # Flag if variance > 10%
    
    print("\n2️⃣  PRICING CONSISTENCY ALERTS (High Variance)")
    if not high_variance.empty:
        print(high_variance[['mean', 'min', 'max', 'CV']].sort_values('CV', ascending=False))
    else:
        print("   No significant pricing anomalies found.")

    # --- INSIGHT 3: MODIFIER IMPACT (Contextual Pricing) ---
    # Does 'TB' (340B) actually lower the price?
    print("\n3️⃣  MODIFIER PRICE IMPACT (Context)")
    mod_impact = df.groupby(['hcpccode', 'modifier_signature'])['unit_cost'].mean().reset_index()
    
    # Pivot to see base price vs modified price side-by-side
    # Identify codes that appear with different modifiers
    dup_codes = mod_impact[mod_impact.duplicated('hcpccode', keep=False)]
    if not dup_codes.empty:
        print(dup_codes.sort_values(['hcpccode', 'modifier_signature']))
    else:
        print("   No differential pricing by modifier found.")

    # --- INSIGHT 4: WASTAGE (JW) LEAKAGE ---
    jw_spend = df[df['modifier_signature'].str.contains('JW')]['paidamount'].sum()
    total_spend = df['paidamount'].sum()
    
    print("\n4️⃣  WASTAGE REPORT (JW Modifier)")
    print(f"   Total Spend on Discarded Drugs: ${jw_spend:,.2f}")
    print(f"   % of Total Spend: {(jw_spend/total_spend)*100:.2f}%")
    
    # --- INSIGHT 5: THE "MARKUP" TRAP (OON Detection) ---
    # Who is charging 10x what we pay? (Likely Out of Network)
    high_markup = df[df['markup_ratio'] > 5.0] # 500% Markup
    if not high_markup.empty:
        print(f"\n5️⃣  EGREGIOUS MARKUPS (>500% Charge/Paid Ratio)")
        print(high_markup[['claimnumber', 'hcpccode', 'paidamount', 'chargedamount', 'markup_ratio']])

# Run the Analysis
generate_insights(df)
