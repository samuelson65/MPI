import pandas as pd
import numpy as np
from Bio import Entrez
from datetime import datetime

# ==========================================
# 1. SETUP: BIOPYTHON EVIDENCE FETCHING
# ==========================================
# This function proves to leadership (and providers) that your limits are based on science.
def fetch_clinical_evidence(term):
    """
    Uses Biopython to search PubMed and return a relevant citation abstract.
    This replaces "Because I said so" with "Because the FDA/CDC says so."
    """
    Entrez.email = "audit.team@yourcompany.com" # Required by NCBI
    try:
        # 1. Search PubMed
        handle = Entrez.esearch(db="pubmed", term=term, retmax=1)
        record = Entrez.read(handle)
        id_list = record["IdList"]
        handle.close()

        if not id_list:
            return "Citation Not Found in PubMed."

        # 2. Fetch Abstract
        handle = Entrez.efetch(db="pubmed", id=id_list[0], rettype="abstract", retmode="text")
        abstract = handle.read()
        handle.close()
        
        # Return a clean snippet (First 300 chars)
        return f"PUBMED ID: {id_list[0]} | SOURCE: {abstract[:200]}..."
    except Exception as e:
        return f"Evidence Fetch Failed: {str(e)}"

# ==========================================
# 2. THE "TRUTH" TABLE (Clinical Knowledge Base)
# ==========================================
# This is where we fix the "Unit Conversion" and "BSA" errors.
CLINICAL_KB = {
    'J9035': { # Avastin
        'name': 'Avastin (Bevacizumab)',
        'mg_per_unit': 10.0,      # FIX #1: 1 Unit = 10 mg (Not 1mg!)
        'dosing_model': 'WEIGHT', # mg/kg
        'limit_factor': 15.0,     # Max 15 mg/kg
        'bio_hard_limit': 160.0,  # Max Weight (kg) - CDC 99th %ile
        'citation_query': "Bevacizumab dosing weight limit oncology"
    },
    'J9190': { # 5-FU
        'name': 'Fluorouracil (5-FU)',
        'mg_per_unit': 10.0,
        'dosing_model': 'BSA',    # FIX #3: Dosed by Surface Area (mg/m2)
        'limit_factor': 2600.0,   # Max 2600 mg/m2
        'bio_hard_limit': 3.0,    # Max BSA (m2) - Statistical Ceiling
        'citation_query': "Body Surface Area distribution cancer patients"
    },
    'J9271': { # Keytruda
        'name': 'Keytruda (Pembrolizumab)',
        'mg_per_unit': 1.0,
        'dosing_model': 'FLAT',   # Fixed Dose
        'limit_factor': 0,        # N/A for flat
        'flat_max_mg': 400.0,     # Hard Cap (mg)
        'citation_query': "Pembrolizumab max flat dose label"
    }
}

# ==========================================
# 3. MOCK DATA (With "Traps")
# ==========================================
data = {
    'claim_id': ['CLM001', 'CLM002', 'CLM003', 'CLM004'],
    'hcpccode': ['J9035', 'J9190', 'J9190', 'J9271'], # Avastin, 5-FU, 5-FU, Keytruda
    'mod1': ['None', 'None', 'JW', 'None'], # Note the JW on CLM003
    'mod2': ['None', 'None', 'None', 'None'],
    'units': [60, 5000, 500, 600], # 60 units of Avastin, 5000 units of 5-FU
    'date_from': ['2024-01-01', '2024-02-01', '2024-02-01', '2024-03-01'],
    'date_to':   ['2024-01-01', '2024-02-05', '2024-02-05', '2024-03-01'], # CLM002 is 5 days!
    'paid_amount': [600, 5000, 500, 30000]
}
df = pd.DataFrame(data)

# ==========================================
# 4. THE BULLETPROOF AUDIT ENGINE
# ==========================================
def run_clinical_audit(df):
    results = []
    
    print("🏥 STARTING CLINICAL AUDIT...\n")
    
    # Process Row by Row (for clarity, vectorization is faster for production)
    for _, row in df.iterrows():
        code = row['hcpccode']
        
        # 0. Skip if unknown drug
        if code not in CLINICAL_KB:
            continue
            
        kb = CLINICAL_KB[code]
        
        # 1. FIX DURATION (The "Overdose" Trap)
        d1 = datetime.strptime(row['date_from'], "%Y-%m-%d")
        d2 = datetime.strptime(row['date_to'], "%Y-%m-%d")
        days = (d2 - d1).days + 1
        
        # 2. FILTER MODIFIERS (The "Waste" Trap)
        is_waste = 'JW' in [row['mod1'], row['mod2']]
        if is_waste:
            # We skip 'Weight' checks on waste lines (Audit separately for vial size)
            results.append({**row, 'audit_status': 'PASS (Waste Line)', 'reason': 'JW Modifier Present'})
            continue

        # 3. NORMALIZE UNITS (The "10x" Trap)
        total_mg_billed = row['units'] * kb['mg_per_unit']
        
        # 4. CALCULATE DAILY DOSE
        daily_mg = total_mg_billed / days
        
        # 5. REVERSE ENGINEER THE PATIENT
        implied_metric = 0
        status = "PASS"
        reason = "Within Norms"
        evidence = ""
        
        if kb['dosing_model'] == 'FLAT':
            if daily_mg > kb['flat_max_mg']:
                status = "DENY"
                reason = f"Exceeds Max Flat Dose ({kb['flat_max_mg']} mg)"
                
        elif kb['dosing_model'] == 'WEIGHT':
            implied_weight = daily_mg / kb['limit_factor']
            implied_metric = implied_weight
            if implied_weight > kb['bio_hard_limit']:
                status = "DENY"
                reason = f"Implied Weight {implied_weight:.1f}kg > Limit {kb['bio_hard_limit']}kg"
        
        elif kb['dosing_model'] == 'BSA':
            implied_bsa = daily_mg / kb['limit_factor']
            implied_metric = implied_bsa
            if implied_bsa > kb['bio_hard_limit']:
                status = "DENY"
                reason = f"Implied BSA {implied_bsa:.2f}m2 > Limit {kb['bio_hard_limit']}m2"

        # 6. FETCH EVIDENCE (Biopython Magic)
        if status == "DENY":
            print(f"   >>> Fetching Evidence for {kb['name']} denial...")
            evidence = fetch_clinical_evidence(kb['citation_query'])

        results.append({
            'claim_id': row['claim_id'],
            'drug': kb['name'],
            'daily_dose_mg': daily_mg,
            'duration_days': days,
            'implied_patient_metric': implied_metric,
            'audit_status': status,
            'reason': reason,
            'clinical_evidence': evidence
        })

    return pd.DataFrame(results)

# ==========================================
# 5. EXECUTE & VIEW
# ==========================================
audit_report = run_clinical_audit(df)

# Display specific columns for the "Director View"
cols = ['claim_id', 'drug', 'duration_days', 'daily_dose_mg', 'audit_status', 'reason', 'clinical_evidence']
print("\n" + "="*60)
print("📄 FINAL AUDIT REPORT")
print("="*60)
print(audit_report[cols].to_string(index=False))
