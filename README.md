import datetime
from Bio import Entrez

# 1. SETUP
Entrez.email = "auditor@insurance.com"  # Required by NCBI

# 2. THE KNOWLEDGE BASE (Simplified for Prototype)
# In production, this comes from OpenFDA or your SQL DB
DRUG_RULES = {
    "J9271": {
        "name": "Keytruda",
        "banned_diagnoses": ["C61", "C71.9"], # Prostate, Glioblastoma (Known failures)
        "approved_keywords": ["Lung", "Melanoma", "Head and Neck", "Hodgkin"]
    },
    "J2506": {
        "name": "Neulasta",
        "rule_type": "TIMING",
        "min_days_after_chemo": 14
    }
}

# 3. THE LOGIC ENGINES

def check_keytruda_indication(diagnosis_code, diagnosis_desc):
    """
    Checks if Keytruda is being used for a 'Banned' cancer type.
    """
    print(f"   [Logic] Checking Indication for {diagnosis_desc} ({diagnosis_code})...")
    
    # A. Hard Fail Check (The "Trap" List)
    if diagnosis_code in DRUG_RULES["J9271"]["banned_diagnoses"]:
        return "❌ DENY: Diagnosis is clinically unsupported (Failed Phase 3 Trials)."
    
    # B. Biopython Check (The "Safety Net")
    # If it's not explicitly banned, ask PubMed if it's approved.
    try:
        query = f'(Pembrolizumab) AND ({diagnosis_desc}) AND ("Phase III" OR "NCCN Category 1")'
        handle = Entrez.esearch(db="pubmed", term=query, retmax=1)
        record = Entrez.read(handle)
        
        if int(record["Count"]) == 0:
            return "⚠️ FLAG: No Phase III evidence found in PubMed. Potential Off-Label."
        else:
            return "✅ APPROVE: Valid Phase III Evidence found."
            
    except Exception as e:
        return f"Error: {e}"

def check_neulasta_timing(neulasta_date, last_chemo_date):
    """
    Enforces the '14-Day Rule' between Chemo and Neulasta.
    """
    print(f"   [Logic] Checking 14-Day Safety Rule...")
    
    d1 = datetime.datetime.strptime(neulasta_date, "%Y-%m-%d")
    d2 = datetime.datetime.strptime(last_chemo_date, "%Y-%m-%d")
    
    delta = (d1 - d2).days
    
    if 0 <= delta < 14:
        return f"❌ DENY: Patient Safety Risk. Administered {delta} days after Chemo (Requires 14+)."
    else:
        return "✅ APPROVE: Timing is safe."

# 4. MAIN AUDIT RUNNER

def audit_claim(claim_data):
    j_code = claim_data["j_code"]
    print(f"\n--- AUDITING CLAIM: {j_code} ({DRUG_RULES.get(j_code, {}).get('name', 'Unknown')}) ---")
    
    if j_code == "J9271": # Keytruda Logic
        result = check_keytruda_indication(claim_data["diag_code"], claim_data["diag_desc"])
        print(f"RESULT: {result}")
        
    elif j_code == "J2506": # Neulasta Logic
        result = check_neulasta_timing(claim_data["date_of_service"], claim_data["last_chemo_date"])
        print(f"RESULT: {result}")

# --- TEST DATA (The "Honey Pot") ---

batch_claims = [
    # Claim 1: Keytruda for Prostate Cancer (Should Deny - High Value Hit)
    {
        "j_code": "J9271",
        "diag_code": "C61",
        "diag_desc": "Malignant Neoplasm of Prostate",
        "date_of_service": "2024-01-10"
    },
    # Claim 2: Neulasta given 2 days after Chemo (Should Deny - Patient Safety)
    {
        "j_code": "J2506", 
        "last_chemo_date": "2024-02-01",
        "date_of_service": "2024-02-03", # Only 2 days gap!
        "diag_desc": "Breast Cancer"
    }
]

# Run the Batch
for claim in batch_claims:
    audit_claim(claim)
