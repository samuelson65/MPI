import requests
from datetime import datetime

def check_zombie_drug(ndc_code, service_date_str):
    """
    Checks if an NDC was Deactivated/Discontinued BEFORE the service date.
    """
    print(f"🧟 Checking NDC: '{ndc_code}' for service on {service_date_str}...")

    # OpenFDA NDC Query
    # Note: NDC format in APIs usually requires dashes (e.g., 0000-0000-00)
    url = f"https://api.fda.gov/drug/ndc.json?search=product_ndc:\"{ndc_code}\"&limit=1"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if "error" in data:
            return "⚠️ Drug Not Found (Check NDC format)."

        result = data['results'][0]
        
        # Check for 'marketing_end_date'
        # If this field is MISSING, the drug is still active (Good).
        end_date_str = result.get('marketing_end_date')
        
        if not end_date_str:
            return "✅ ACTIVE: Drug is currently on the market."

        # Parse Dates
        service_date = datetime.strptime(service_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y%m%d")

        print(f"   --> Drug was Discontinued on: {end_date.strftime('%Y-%m-%d')}")

        # The Logic
        if service_date > end_date:
            days_late = (service_date - end_date).days
            return f"❌ DENY: Zombie Billing. Drug was discontinued {days_late} days BEFORE this service."
        else:
            return "✅ PASS: Service date is valid (Before discontinuation)."

    except Exception as e:
        return f"Error: {e}"

# --- TEST ---

# 1. Test a known discontinued NDC (hypothetical example)
# Let's say this NDC expired in 2019, but we bill it in 2024
print(check_zombie_drug("0029-6058", "2024-01-01")) 


import pandas as pd
import datetime

# ==========================================
# 1. THE "CITATION" KNOWLEDGE BASE
# ==========================================
# This dictionary maps simple Rules to specific Legal/Clinical text.
# You print this "Citation" on the denial letter so the provider cannot appeal.

ONCOLOGY_RULES_DB = {
    # --- DRUG 1: KEYTRUDA (Pembrolizumab) ---
    "J9271": {
        "drug_name": "Keytruda",
        "denial_citations": {
            "C61": { # Prostate Cancer
                "status": "DENY",
                "reason": "Experimental / Failed Clinical Trials",
                "citation": "Merck Press Release (01/25/2023): KEYNOTE-991 trial for mHSPC stopped for futility. No improvement in OS/rPFS."
            },
            "C71.9": { # Glioblastoma
                "status": "DENY",
                "reason": "Experimental / Failed Clinical Trials",
                "citation": "Nature Medicine (2019): KEYNOTE-028 showed limited efficacy. Not NCCN Category 1/2A recommended."
            }
        },
        "approved_indications_ref": "FDA Label Section 1.1-1.18 (Revised 2024)"
    },

    # --- DRUG 2: OPDIVO (Nivolumab) ---
    "J9299": {
        "drug_name": "Opdivo",
        "regimen_rules": {
            "C64": { # Renal Cell Carcinoma (Kidney)
                "required_combo": "J9480", # Yervoy
                "citation": "FDA Approval (04/16/2018): Based on CheckMate-214 trial. Indicated for intermediate/poor risk RCC *in combination with ipilimumab*."
            }
        }
    },

    # --- DRUG 3: NEULASTA (Pegfilgrastim) ---
    "J2506": {
        "drug_name": "Neulasta",
        "timing_rule": {
            "min_gap_days": 14,
            "citation": "FDA Prescribing Information, Section 2.1: 'Do not administer Neulasta between 14 days before and 24 hours after administration of cytotoxic chemotherapy.'"
        }
    },
    
    # --- DRUG 4: RITUXAN (Rituximab) ---
    "J9312": {
        "drug_name": "Rituxan",
        "coding_rule": {
            "unlisted_code": "J3590",
            "citation": "CMS HCPCS Level II Guidelines: 'If a specific code describes a service (J9312), the unlisted code (J3590) must not be reported.'"
        }
    }
}

# ==========================================
# 2. THE WASTAGE RULES (CMS Chapter 17)
# ==========================================
WASTE_POLICY_DB = {
    "MDV_VIALS": {
        # List of Multi-Dose NDCs (Sample)
        "00002-7510-01": "Humalog", 
        "50242-053-06": "Rituxan (MDV Version)"
    },
    "citation": "CMS Medicare Claims Processing Manual, Chapter 17, Section 40: 'The JW modifier is not appropriate for drugs that are from multiple dose vials.'"
}

# ==========================================
# 3. THE AUDIT LOGIC
# ==========================================

def audit_claim_with_citations(claim):
    """
    Input: Dictionary containing claim details.
    Output: Audit Result with CITATION string.
    """
    j_code = claim.get("j_code")
    diagnosis = claim.get("diagnosis", "")
    
    print(f"\n📝 Auditing {j_code} for Diagnosis {diagnosis}...")

    # --- CHECK 1: INDICATION & REGIMEN (Keytruda/Opdivo) ---
    if j_code in ONCOLOGY_RULES_DB:
        rule = ONCOLOGY_RULES_DB[j_code]
        
        # A. Keytruda Specific Denials
        if "denial_citations" in rule:
            if diagnosis in rule["denial_citations"]:
                denial = rule["denial_citations"][diagnosis]
                return f"❌ DENY: {denial['reason']}\n   >> AUTHORITY: {denial['citation']}"

        # B. Opdivo Regimen Check
        if "regimen_rules" in rule:
            # Check if diagnosis matches a mandatory combo rule
            # (Using 'starts with' for ICD-10 variations like C64.1)
            for diag_prefix, requirement in rule["regimen_rules"].items():
                if diagnosis.startswith(diag_prefix):
                    required_drug = requirement["required_combo"]
                    # In a real script, we'd check the whole claim line items here
                    if required_drug not in claim.get("all_codes_on_claim", []):
                        return f"❌ DENY: Missing Mandated Combination ({required_drug})\n   >> AUTHORITY: {requirement['citation']}"

    # --- CHECK 2: TIMING (Neulasta) ---
    if j_code == "J2506": # Neulasta
        date_service = datetime.datetime.strptime(claim["date_service"], "%Y-%m-%d")
        date_chemo = datetime.datetime.strptime(claim["last_chemo_date"], "%Y-%m-%d")
        days_gap = (date_service - date_chemo).days
        
        limit = ONCOLOGY_RULES_DB["J2506"]["timing_rule"]["min_gap_days"]
        if days_gap < limit:
            citation = ONCOLOGY_RULES_DB["J2506"]["timing_rule"]["citation"]
            return f"❌ DENY: Administered {days_gap} days after chemo (Limit: {limit}+).\n   >> AUTHORITY: {citation}"

    # --- CHECK 3: WASTAGE INTEGRITY (Rituxan/Others) ---
    if claim.get("modifier") == "JW":
        ndc = claim.get("ndc")
        if ndc in WASTE_POLICY_DB["MDV_VIALS"]:
            citation = WASTE_POLICY_DB["citation"]
            return f"❌ DENY: Waste billed on Multi-Dose Vial.\n   >> AUTHORITY: {citation}"

    return "✅ APPROVE: No audit flags found."

# ==========================================
# 4. TEST SCENARIOS
# ==========================================

# Scenario A: Keytruda for Prostate Cancer (Failed Trial)
claim_1 = {
    "j_code": "J9271", 
    "diagnosis": "C61", 
    "date_service": "2024-01-01"
}

# Scenario B: Opdivo for Kidney Cancer, but missing Yervoy (Regimen Fail)
claim_2 = {
    "j_code": "J9299", 
    "diagnosis": "C64.9", 
    "all_codes_on_claim": ["J9299"], # Yervoy (J9480) is missing!
    "date_service": "2024-01-01"
}

# Scenario C: Neulasta given 3 days after Chemo (Safety Fail)
claim_3 = {
    "j_code": "J2506", 
    "diagnosis": "C50.9",
    "date_service": "2024-02-04",
    "last_chemo_date": "2024-02-01" # Only 3 days gap
}

print(audit_claim_with_citations(claim_1))
print(audit_claim_with_citations(claim_2))
print(audit_claim_with_citations(claim_3))
