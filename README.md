import pandas as pd
import numpy as np

# ==========================================
# 1. GENERATE SAMPLE DATA (Replace this with your pd.read_csv)
# ==========================================
data = {
    'claimnumber': [f'CLM{i}' for i in range(1001, 1011)],
    'hcpccode': ['J9271', 'J9271', 'J9271', 'J0896', 'J0896', 'L0650', 'L0650', '99213', '99213', '99213'],
    # Modifiers: Some empty (NaN), some with high-impact mods like JW, 25, TB
    'mod1': ['None', 'JW', 'None', 'JZ', 'JZ', 'NU', 'NU', '25', 'None', '25'],
    'mod2': [None, None, None, 'JB', 'JB', 'RT', 'LT', None, None, None],
    'mod3': [None, None, None, 'TB', None, None, None, None, None, None],
    'mod4': [None, None, None, 'PO', None, None, None, None, None, None],
    'mod5': [None, None, None, None, None, None, None, None, None, None],
    'paidamount': [5000, 500, 5000, 1500, 2000, 800, 800, 150, 100, 150], # Note variance in J0896 ($1500 vs $2000)
    'chargedamount': [12000, 1200, 12000, 5000, 6000, 2500, 2500, 300, 200, 300],
    'date_of_service': pd.date_range(start='1/1/2024', periods=10),
    'unitcount': [10, 1, 10, 50, 50, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# ==========================================
# 2. DATA PRE-PROCESSING (The "Explosion")
# ==========================================
def preprocess_claims(df):
    # Combine all modifier columns into a single "Signature" for row-level context
    # This helps distinct "J9271" from "J9271 with JW"
    mod_cols = ['mod1', 'mod2', 'mod3', 'mod4', 'mod5']
    
    # Create a clean list of modifiers for each row (removing None/NaN)
    df['modifier_signature'] = df[mod_cols].apply(
        lambda x: ','.join(x.dropna().astype(str)), axis=1
    )
    
    # Calculate Unit Cost (Critical for variance analysis)
    df['unit_cost'] = df['paidamount'] / df['unitcount'].replace(0, 1)
    
    return df

df = preprocess_claims(df)

# ==========================================
# 3. ANALYSIS: TOP DRUGS BY SPEND
# ==========================================
def analyze_top_drugs(df):
    print("\n🏆 TOP 5 DRUGS BY TOTAL SPEND")
    print("-" * 50)
    
    # Group by Code
    stats = df.groupby('hcpccode').agg(
        total_spend=('paidamount', 'sum'),
        total_units=('unitcount', 'sum'),
        avg_unit_cost=('unit_cost', 'mean'),
        claim_count=('claimnumber', 'count')
    ).sort_values('total_spend', ascending=False).head(5)
    
    print(stats)
    return stats

# ==========================================
# 4. ANALYSIS: THE MODIFIER IMPACT (The "Insight")
# ==========================================
def analyze_modifier_context(df):
    print("\n🔍 MODIFIER IMPACT ANALYSIS")
    print("-" * 50)
    
    # A. WASTAGE ANALYSIS (JW Modifier)
    # Filter rows where ANY modifier column contains 'JW'
    waste_mask = df['modifier_signature'].str.contains('JW')
    waste_spend = df[waste_mask]['paidamount'].sum()
    total_spend = df['paidamount'].sum()
    
    print(f"🗑️  Total Wastage Spend (JW): ${waste_spend:,.2f}")
    print(f"📊 % of Spend Wasted: {(waste_spend/total_spend)*100:.1f}%")
    
    # B. PRICING VARIANCE (Did Modifiers change the price?)
    # We group by Code AND Modifier Signature to see price tiers
    print("\n📉 UNIT COST VARIANCE BY MODIFIER (Are you paying more for specific mods?)")
    variance = df.groupby(['hcpccode', 'modifier_signature']).agg(
        avg_unit_cost=('unit_cost', 'mean'),
        total_paid=('paidamount', 'sum'),
        claims=('claimnumber', 'count')
    ).reset_index()
    
    # Filter for interesting variances (e.g., J0896)
    variance['hcpccode'] = variance['hcpccode'].astype(str)
    print(variance.sort_values(['hcpccode', 'avg_unit_cost']))

    return variance

# ==========================================
# 5. ANALYSIS: SPECIALTY FLAGS (TB, PO, 25)
# ==========================================
def analyze_red_flags(df):
    print("\n🚩 RED FLAG REPORT")
    print("-" * 50)
    
    # 340B Discount Check (TB Modifier)
    tb_claims = df[df['modifier_signature'].str.contains('TB')]
    if not tb_claims.empty:
        print(f"⚠️  340B CLAIMS DETECTED: {len(tb_claims)} claims.")
        print(f"    Avg Unit Cost for TB: ${tb_claims['unit_cost'].mean():.2f}")
        print("    (Action: Verify these were paid at the lower 340B rate)")
        
    # Unbundling Check (Modifier 25 on E&M Codes)
    em_codes = df[df['hcpccode'].str.startswith('99')] # Office visits
    mod_25_count = em_codes['modifier_signature'].str.contains('25').sum()
    total_em = len(em_codes)
    
    if total_em > 0:
        print(f"\n⚠️  MODIFIER 25 USAGE RATE: {(mod_25_count/total_em)*100:.1f}% of Office Visits")
        print("    (Benchmark: If > 50%, audit for unbundling)")

# ==========================================
# EXECUTE
# ==========================================
analyze_top_drugs(df)
variance_df = analyze_modifier_context(df)
analyze_red_flags(df)
