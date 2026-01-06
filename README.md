import pandas as pd

# --- KNOWLEDGE BASE (The "Brain") ---

# 1. REGIMEN RULES: "If you have Key, you MUST have Value"
REGIMEN_MANDATES = {
    "J9299": { # Opdivo
        "required_combinations": {
            "C64": "J9480" # If Renal Cancer (C64), require Yervoy (J9480)
        }
    },
    "J9041": { # Velcade
        "required_combinations": {
            "C90": "J9171" # If Multiple Myeloma, often requires Dexamethasone/Revlimid
        }
    }
}

# 2. VIAL DATABASE: "Is this drug a Multi-Dose Vial?"
# (In prod, fetch this from OpenFDA NDC directory)
NDC_METADATA = {
    "50242-051-21": {"drug": "Rituxan", "type": "SDV", "size_mg": 100}, # Single Dose (Waste OK)
    "00002-7510-01": {"drug": "Humalog", "type": "MDV", "size_ml": 10}, # Multi Dose (Waste ILLEGAL)
}

# --- THE ADVANCED LOGIC ---

def audit_claim_advanced(claim_df):
    """
    Input: Pandas DataFrame representing ONE claim (multiple lines).
    """
    audits = []
    
    # Get all J-Codes and Diagnosis on this claim
    all_j_codes = set(claim_df['j_code'].unique())
    primary_diag = claim_df['diagnosis'].iloc[0] # Assuming claim-level diag
    
    print(f"🕵️ Auditing Claim with codes: {all_j_codes} for {primary_diag}")

    # --- CHECK 1: REGIMEN COMPLETENESS ---
    for j_code in all_j_codes:
        if j_code in REGIMEN_MANDATES:
            rule = REGIMEN_MANDATES[j_code]
            
            # Check if this diagnosis triggers a mandated combo
            # (Using 'startswith' to handle ICD-10 sub-codes like C64.1)
            matched_diag = next((d for d in rule["required_combinations"] if primary_diag.startswith(d)), None)
            
            if matched_diag:
                required_drug = rule["required_combinations"][matched_diag]
                if required_drug not in all_j_codes:
                    audits.append({
                        "status": "DENY",
                        "code": j_code,
                        "reason": f"Regimen Failure. {j_code} for {primary_diag} requires {required_drug}, but it is missing."
                    })

    # --- CHECK 2: WASTAGE (JW) INTEGRITY ---
    # Filter for lines with 'JW' modifier
    wastage_lines = claim_df[claim_df['modifier'] == 'JW']
    
    for _, row in wastage_lines.iterrows():
        ndc = row['ndc']
        if ndc in NDC_METADATA:
            vial_info = NDC_METADATA[ndc]
            
            # A. The "Multi-Dose" Check
            if vial_info['type'] == 'MDV':
                audits.append({
                    "status": "DENY",
                    "code": row['j_code'],
                    "reason": f"Illegal Wastage. {ndc} is a Multi-Dose Vial (MDV). Waste (JW) is never payable."
                })
                
            # B. The "Impossible Waste" Check (Math)
            # Example: Vial is 100mg. Doctor billed 110mg waste. Impossible.
            # (Simplified logic for demo)
            billed_waste_units = row['units']
            if billed_waste_units > vial_info['size_mg']: # Assuming 1 unit = 1 mg
                 audits.append({
                    "status": "DENY",
                    "code": row['j_code'],
                    "reason": f"Impossible Waste. Billed {billed_waste_units} units waste, but vial is only {vial_info['size_mg']} units."
                })

    return audits

# --- TEST DATA ---

# Scenario: Renal Cancer patient. Doctor billed Opdivo (J9299) but forgot Yervoy.
# Also billed waste on an Insulin MDV (illegal).
claim_data = pd.DataFrame([
    {"line": 1, "j_code": "J9299", "diagnosis": "C64.9", "modifier": "None", "ndc": "99999-999-99", "units": 240},
    {"line": 2, "j_code": "J1817", "diagnosis": "E11.9", "modifier": "JW", "ndc": "00002-7510-01", "units": 5} 
])

results = audit_claim_advanced(claim_data)

for r in results:
    print(f"🚨 {r['status']}: {r['reason']}")
