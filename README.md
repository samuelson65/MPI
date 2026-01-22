import pandas as pd
import numpy as np
from pyspark.sql import SparkSession

# ==========================================
# 1. SAMPLE DATA LOADING (Messy Data)
# ==========================================
raw_data = {
    'claimnumber': [f'CLM{i}' for i in range(1001, 1021)],
    'hcpccode': [
        'J9271', 'J9271', 'J9271', 'J9271',  # Keytruda
        'J0896', 'J0896', 'J0896', 'J0896',  # Reblozyl
        'L0650', 'L0650', 'L0650',           # DME
        '99213', '99213', '99213', '99213',  # Office Visits
        'J9999', 'J9999', 'J3490', 'J3590'   # Unlisted
    ],
    'mod1': ['None', 'JW', 'None', 'None', 'JZ', 'JZ', 'JZ', 'JZ', 'NU', 'NU', 'NU', '25', 'None', '25', 'None', None, None, 'JW', None],
    'mod2': [None, None, None, None, 'JB', 'JB', 'JB', 'JB', 'RT', 'LT', 'RT', None, None, None, None, None, None, None, None],
    'mod3': [None, None, None, None, 'TB', None, 'TB', None, None, None, None, None, None, None, None, None, None, None, None],
    'mod4': [None, None, None, None, 'PO', None, 'PO', None, None, None, None, None, None, None, None, None, None, None, None],
    'mod5': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    'paidamount': ['$5,000.00', '500', 5000, 5100, 1500, 2000, 1500, 2000, 800, 800, 850, 150, 100, 150, 100, 3000, 3200, 400, 150],
    'chargedamount': ['12000', 1200, 12000, 12500, 5000, 6000, 5000, 6000, 2500, 2500, 2600, 300, 200, 300, 200, 9000, 9500, 1200, 5000],
    'date_of_service': pd.date_range(start='1/1/2024', periods=19).tolist() + ['2024-05-01'],
    'unitcount': ['10', 1, 10, 10, 50, 50, 50, 50, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

# Ensure lists are same length for dataframe
max_len = max(len(v) for v in raw_data.values())
for k in raw_data:
    if len(raw_data[k]) < max_len: raw_data[k] = raw_data[k] * (max_len // len(raw_data[k]) + 1)
    raw_data[k] = raw_data[k][:max_len]

df = pd.DataFrame(raw_data)

# ==========================================
# 2. CLEANING ENGINE
# ==========================================
def clean_data(df):
    mod_cols = ['mod1', 'mod2', 'mod3', 'mod4', 'mod5']
    for col in mod_cols:
        df[col] = df[col].fillna('').astype(str).str.upper().str.replace('NONE', '').str.replace('NAN', '')
    df['modifier_signature'] = df[mod_cols].apply(lambda x: ','.join(filter(None, x)), axis=1)

    for col in ['paidamount', 'chargedamount', 'unitcount']:
        df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['unit_cost'] = df['paidamount'] / df['unitcount'].replace(0, 1)
    return df

df_clean = clean_data(df)

# ==========================================
# 3. OUTPUT GENERATION (Use display() for Download)
# ==========================================

print("---------------------------------------------------------")
print("📥 REPORT 1: TOP SPEND DRIVERS (Pareto Chart Data)")
print("   ACTION: Click the arrow icon below to Download CSV")
print("---------------------------------------------------------")
insight1 = df_clean.groupby('hcpccode').agg(
    Total_Spend=('paidamount', 'sum'),
    Total_Units=('unitcount', 'sum'),
    Claim_Count=('claimnumber', 'count')
).sort_values('Total_Spend', ascending=False).reset_index()
display(insight1) # <--- Native Databricks Table


print("\n---------------------------------------------------------")
print("📥 REPORT 2: PRICING VARIANCE (Contract Errors)")
print("   ACTION: Look for High CoV (>0.1)")
print("---------------------------------------------------------")
insight2 = df_clean.groupby('hcpccode')['unit_cost'].agg(['mean', 'std', 'min', 'max'])
insight2['CoV'] = insight2['std'] / insight2['mean']
insight2 = insight2[insight2['CoV'] > 0.05].sort_values('CoV', ascending=False).reset_index()
display(insight2)


print("\n---------------------------------------------------------")
print("📥 REPORT 3: MODIFIER IMPACT (340B / Site of Care)")
print("---------------------------------------------------------")
insight3 = df_clean.groupby(['hcpccode', 'modifier_signature']).agg(
    Avg_Price=('unit_cost', 'mean'),
    Total_Paid=('paidamount', 'sum'),
    Count=('claimnumber', 'count')
).reset_index()
display(insight3)


print("\n---------------------------------------------------------")
print("📥 REPORT 4: WASTAGE LOG (JW Modifier Analysis)")
print("---------------------------------------------------------")
insight4 = df_clean[df_clean['modifier_signature'].str.contains('JW')][['claimnumber', 'hcpccode', 'paidamount', 'date_of_service', 'modifier_signature']]
display(insight4)


print("\n---------------------------------------------------------")
print("📥 REPORT 5: UNLISTED CODES (High Risk Audit)")
print("---------------------------------------------------------")
insight5 = df_clean[df_clean['hcpccode'].isin(['J3490', 'J3590', 'J9999'])][['claimnumber', 'hcpccode', 'paidamount', 'date_of_service']]
display(insight5)
