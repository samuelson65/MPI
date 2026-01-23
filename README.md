import pandas as pd
from Bio import Entrez
import numpy as np

# ==========================================
# 1. THE "SAFE HARBOR" KNOWLEDGE BASE
# ==========================================
CLINICAL_KB = {
    # --- SAFE TO AUDIT (Single Session) ---
    'J9271': { 
        'name': 'Keytruda (Pembrolizumab)',
        'mg_per_unit': 1.0,
        'dosing_model': 'FLAT',
        'limit_factor': 0,
        'flat_max_mg': 400.0,
        'risk_profile': 'SINGLE_SESSION', # SAFE: Given in ~30 mins
        'citation_query': "Pembrolizumab max flat dose label"
    },
    'J9035': { 
        'name': 'Avastin (Bevacizumab)',
        'mg_per_unit': 10.0,
        'dosing_model': 'WEIGHT',
        'limit_factor': 15.0,     # Max 15 mg/kg
        'bio_hard_limit': 160.0,  # Max Weight (kg)
        'risk_profile': 'SINGLE_SESSION', # SAFE: Given in ~90 mins
        'citation_query': "Bevacizumab dosing weight limit oncology"
    },
    
    # --- UNSAFE TO AUDIT (Continuous Infusion) ---
    'J9190': { 
        'name': 'Fluorouracil (5-FU)',
        'mg_per_unit': 10.0,
        'dosing_model': 'BSA',
        'limit_factor': 2600.0,
        'bio_hard_limit': 3.0,
        'risk_profile': 'CONTINUOUS_INFUSION', # UNSAFE: Often 46-hour pump
        'citation_query': "Fluorouracil max daily dose"
    }
}

# ==========================================
# 2. MOCK DATA (Missing Date_To)
# ==========================================
data = {
    'claim_id': ['CLM001', 'CLM002', 'CLM003'],
    'hcpccode': ['J9271', 'J9190', 'J9035'], 
    'paid_amount': [500, 500, 500],
    'units': [
        600,  # Keytruda: 600mg (Limit is 400mg) -> SHOULD FLAG
        5000, # 5-FU: 50,000mg (Could be 5 days) -> SHOULD SKIP
        50    # Avastin: 500mg (Implies 33kg person) -> PASS
    ],
    'date_of_service': ['2024-01-01', '2024-01-02', '2024-01-03'] 
    # NO date_to column!
}
df = pd.DataFrame(data)

# ==========================================
# 3. THE "DURATION-AGNOSTIC" ENGINE
# ==========================================
def run_safe_audit(df):
    results = []
    print("🛡️  STARTING 'SAFE HARBOR' AUDIT (Handling Missing Duration)...\n")
    
    for _, row in df.iterrows():
        code = row['hcpccode']
        if code not in CLINICAL_KB: continue
            
        kb = CLINICAL_KB[code]
        
        # --- CHECK 1: SAFETY FILTER ---
        # If it's a multi-day drug and we lack dates, ABORT logic for this line.
        if kb['risk_profile'] == 'CONTINUOUS_INFUSION':
            results.append({
                'claim_id': row['claim_id'],
                'drug': kb['name'],
                'audit_status': 'SKIPPED',
                'reason': 'Multi-day drug requires Date Span to audit safely.'
            })
            continue

        # --- CHECK 2: STANDARD LOGIC (For Single Session Drugs) ---
        # Since it's single session, Total Units = Daily Dose
        total_mg = row['units'] * kb['mg_per_unit']
        
        status = "PASS"
        reason = "Within Norms"
        
        # A. FLAT DOSE CHECK
        if kb['dosing_model'] == 'FLAT':
            if total_mg > kb['flat_max_mg']:
                status = "DENY"
                reason = f"Exceeds Max Single Session Dose ({kb['flat_max_mg']} mg)"
        
        # B. WEIGHT CHECK
        elif kb['dosing_model'] == 'WEIGHT':
            implied_weight = total_mg / kb['limit_factor']
            if implied_weight > kb['bio_hard_limit']:
                status = "DENY"
                reason = f"Implied Weight {implied_weight:.1f}kg > Limit {kb['bio_hard_limit']}kg"

        results.append({
            'claim_id': row['claim_id'],
            'drug': kb['name'],
            'audit_status': status,
            'reason': reason
        })
        
    return pd.DataFrame(results)

# Run
report = run_safe_audit(df)
print(report[['claim_id', 'drug', 'audit_status', 'reason']].to_string(index=False))
