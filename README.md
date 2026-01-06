import pandas as pd

# --- KNOWLEDGE BASE ---
# FIXED: Normalized the keys to 'unit_size' instead of mixing 'size_mg'/'size_ml'
NDC_METADATA = {
    "50242-051-21": {"drug": "Rituxan", "type": "SDV", "unit_size": 100, "uom": "mg"}, 
    "00002-7510-01": {"drug": "Humalog", "type": "MDV", "unit_size": 10, "uom": "ml"}, 
}

REGIMEN_MANDATES = {
    "J9299": {"required_combinations": {"C64": "J9480"}}
}

def audit_claim_advanced(claim_df):
    audits = []
    
    # ... (Regimen Logic remains the same) ...

    # --- CHECK 2: WASTAGE (JW) INTEGRITY ---
    wastage_lines = claim_df[claim_df['modifier'] == 'JW']
    
    for _, row in wastage_lines.iterrows():
        ndc = row['ndc']
        
        # Safety Check: Does this NDC exist in our DB?
        if ndc in NDC_METADATA:
            vial_info = NDC_METADATA[ndc]
            
            # A. The "Multi-Dose" Check
            if vial_info['type'] == 'MDV':
                audits.append({
                    "status": "DENY",
                    "code": row['j_code'],
                    "reason": f"Illegal Wastage. {ndc} ({vial_info['drug']}) is a Multi-Dose Vial."
                })
                
            # B. The "Impossible Waste" Check (Math)
            # FIXED: Using the normalized 'unit_size' key
            max_size = vial_info.get('unit_size', 99999) # Default to high number if missing
            billed_waste = row['units']
            
            if billed_waste > max_size:
                 audits.append({
                    "status": "DENY",
                    "code": row['j_code'],
                    "reason": f"Impossible Waste. Billed {billed_waste} units, but vial is only {max_size} {vial_info.get('uom')}."
                })

    return audits

# --- TEST DATA ---
claim_data = pd.DataFrame([
    {"line": 1, "j_code": "J9299", "diagnosis": "C64.9", "modifier": "None", "ndc": "99999-999-99", "units": 240},
    # This line triggered the error before because it accessed 'size_mg' on a 'ml' drug
    {"line": 2, "j_code": "J1817", "diagnosis": "E11.9", "modifier": "JW", "ndc": "00002-7510-01", "units": 15} 
])

results = audit_claim_advanced(claim_data)

for r in results:
    print(f"🚨 {r['status']}: {r['reason']}")
