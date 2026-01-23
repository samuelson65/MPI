import pandas as pd
import numpy as np
from Bio import Entrez

# ==========================================
# 1. SETUP & KNOWLEDGE BASE
# ==========================================
# This dictionary drives the entire audit. 
# It links Codes -> Biological Limits -> Modifier Rules -> Citations.

CLINICAL_KB = {
    'J9271': { # Keytruda
        'name': 'Keytruda (Pembrolizumab)',
        'mg_per_unit': 1.0,
        'model': 'FLAT',
        'limit': 400.0, # Max mg per session
        'risk': 'SINGLE_SESSION',
        'vial_sizes': [50, 100], # Common vials
        'citation': "Pembrolizumab FDA Label Section 2.1 (Dosage)"
    },
    'J9035': { # Avastin
        'name': 'Avastin (Bevacizumab)',
        'mg_per_unit': 10.0, # WATCH OUT: 1 Unit = 10mg
        'model': 'WEIGHT',
        'limit': 15.0, # mg/kg
        'bio_ceiling': 160.0, # Max Weight kg (CDC 99th %ile)
        'risk': 'SINGLE_SESSION',
        'vial_sizes': [100, 400],
        'citation': "Bevacizumab FDA Label; CDC Anthropometric Data 2016"
    },
    'J9190': { # 5-FU (The Trap)
        'name': 'Fluorouracil (5-FU)',
        'mg_per_unit': 10.0,
        'model': 'BSA',
        'limit': 2600.0, # mg/m2
        'risk': 'CONTINUOUS', # UNSAFE without end-date
        'vial_sizes': [500, 1000],
        'citation': "Fluorouracil Dosing Guidelines"
    }
}

# ==========================================
# 2. EVIDENCE FETCHER (Biopython)
# ==========================================
def fetch_evidence_summary(query):
    """
    Simulates fetching a real abstract from PubMed to justify the denial.
    """
    Entrez.email = "audit@yourorg.com"
    # In a real run, you would uncomment the Entrez calls below.
    # For this demo, we return the pre-fetched citation string to ensure speed.
    return f"EVIDENCE SOURCE: PubMed Search for '{query}' confirms dosage limits."

# ==========================================
# 3. MOCK DATA (Complex Modifiers)
# ==========================================
data = {
    'claim_id': ['CLM01', 'CLM02', 'CLM03', 'CLM04', 'CLM05'],
    'hcpccode': ['J9271', 'J9271', 'J9035', 'J9035', 'J9190'],
    
    # MODIFIERS:
    # CLM02: Has JW (Waste) on a separate line
    # CLM04: Has TB (340B Discount)
    'mod1': ['None', 'JW', 'None', 'TB', 'None'], 
    'mod2': ['None', 'None', 'None', 'None', 'None'],
    
    # UNITS & PAYMENTS
    'units': [
        600,  # CLM01: Keytruda 600mg (Limit 400) -> FAIL Clinical
        100,  # CLM02: Keytruda Waste 100mg -> CHECK Vial Integrity
        40,   # CLM03: Avastin 40 units (400mg) -> PASS (40kg person is possible)
        40,   # CLM04: Avastin 40 units (TB Mod) -> CHECK Pricing
        5000  # CLM05: 5-FU Multi-day -> SKIP (Safety)
    ],
    'paid_amount': [
        30000, # CLM01
        5000,  # CLM02 (Waste paid full price)
        2000,  # CLM03 (Standard Price)
        2000,  # CLM04 (TB Claim - Paid same as Standard? -> OVERPAYMENT)
        500   
    ]
}
df = pd.DataFrame(data)

# ==========================================
# 4. THE MULTI-LOGIC AUDIT ENGINE
# ==========================================
def run_director_audit(df):
    audit_log = []
    print("🚀 STARTING MULTI-LOGIC AUDIT...\n")

    for _, row in df.iterrows():
        code = row['hcpccode']
        if code not in CLINICAL_KB: continue
        kb = CLINICAL_KB[code]
        
        # --- PRE-PROCESSING ---
        # 1. Identify Modifiers
        mods = [row['mod1'], row['mod2']]
        is_waste = 'JW' in mods
        is_340b = 'TB' in mods
        
        # 2. Calculate Actual Milligrams
        mg_billed = row['units'] * kb['mg_per_unit']
        
        status = "PASS"
        logic_used = "Standard Review"
        citation = ""

        # --- LOGIC 1: SAFETY FILTER (The "Do No Harm" Rule) ---
        if kb['risk'] == 'CONTINUOUS':
            audit_log.append({
                'Claim': row['claim_id'], 'Drug': kb['name'], 'Result': 'SKIPPED',
                'Logic': 'Safety Filter', 'Details': 'Continuous infusion requires Date Span.'
            })
            continue

        # --- LOGIC 2: WASTE INTEGRITY (JW Modifier) ---
        # Citation: CMS Medicare Claims Processing Manual, Ch 17.
        if is_waste:
            # Rule: You cannot waste more than a full single-use vial.
            # If waste is 100mg, and 50mg vials exist, why open a 100mg vial?
            min_vial = min(kb['vial_sizes'])
            if mg_billed >= min_vial:
                status = "FLAG"
                logic_used = "Waste Optimization (JW)"
                details = f"Waste ({mg_billed}mg) >= Smallest Vial ({min_vial}mg). Provider opened unnecessary vial."
                citation = "CMS Manual Ch 17 Sec 40 (Discarded Drugs)"
            else:
                details = "Waste amount within reasonable limits."
            
            audit_log.append({'Claim': row['claim_id'], 'Drug': kb['name'], 'Result': status, 'Logic': logic_used, 'Details': details, 'Citation': citation})
            continue # Stop here for waste lines

        # --- LOGIC 3: CLINICAL CEILING (Bio-Limits) ---
        # Citation: FDA Labels / CDC Data
        if kb['model'] == 'FLAT':
            if mg_billed > kb['limit']:
                status = "DENY"
                logic_used = "FDA Max Dose"
                details = f"Billed {mg_billed}mg > FDA Max {kb['limit']}mg."
                citation = kb['citation']
                
        elif kb['model'] == 'WEIGHT':
            implied_weight = mg_billed / kb['limit']
            if implied_weight > kb['bio_ceiling']:
                status = "DENY"
                logic_used = "Biological Impossibility"
                details = f"Implied Weight {implied_weight:.1f}kg > CDC Limit {kb['bio_ceiling']}kg."
                citation = kb['citation']

        if status == "DENY":
            audit_log.append({'Claim': row['claim_id'], 'Drug': kb['name'], 'Result': status, 'Logic': logic_used, 'Details': details, 'Citation': citation})
            continue

        # --- LOGIC 4: 340B PRICING CHECK (TB Modifier) ---
        # Citation: HRSA 340B Prime Vendor Program
        if is_340b:
            # Simple heuristic: Unit Cost should be ~30% lower than ASP
            unit_cost = row['paid_amount'] / row['units']
            # Mock Benchmark: Let's say Standard Avastin is $50/unit
            standard_rate = 50.0 
            if unit_cost >= standard_rate:
                status = "OVERPAYMENT"
                logic_used = "340B Discount Missed"
                details = f"Claim has TB modifier but paid full rate (${unit_cost}). Should be discounted."
                citation = "HRSA 340B Pricing Guidelines"
                
                audit_log.append({'Claim': row['claim_id'], 'Drug': kb['name'], 'Result': status, 'Logic': logic_used, 'Details': details, 'Citation': citation})
                continue

        # If it survived all checks
        audit_log.append({'Claim': row['claim_id'], 'Drug': kb['name'], 'Result': 'PASS', 'Logic': 'All Checks', 'Details': 'Within norms'})

    return pd.DataFrame(audit_log)

# Run and Display
report = run_director_audit(df)
print(report.to_string(index=False))
