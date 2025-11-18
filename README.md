import json

# =============================================================================
# 1. DATA LIBRARIES (The "Knowledge Base")
# =============================================================================

# A. DRG Grouper / Mapper
# Maps specific MS-DRG codes to the Clinical Categories used in our guidelines.
# NOTE: In a real system, this would be a database query.
DRG_CATEGORY_MAP = {
    # Heart Failure
    '291': 'CHF', '292': 'CHF', '293': 'CHF',
    # AMI / PCI
    '280': 'AMI', '281': 'AMI', '282': 'AMI',
    '246': 'PCI', '247': 'PCI',
    # Pneumonia / Sepsis
    '193': 'PNA', '194': 'PNA', '195': 'PNA',
    '871': 'SEPSIS', '872': 'SEPSIS',
    # COPD
    '190': 'COPD', '191': 'COPD', '192': 'COPD',
    # Surgery (Orthopedic / General)
    '470': 'TKA_THA', '469': 'TKA_THA', # Knee/Hip replacement
    '329': 'BOWEL_SURGERY', # Major Small & Large Bowel Procedures
    # Complications / Targets
    '682': 'RENAL_FAILURE',
    '544': 'PJI', # Prosthetic Joint Infection
    '999': 'TRAUMA', # Placeholder for unrelated trauma
    '000': 'OTHER'
}

# B. GMLOS Dictionary (Geometric Mean Length of Stay)
# Maps DRG to its benchmark days. (Mock values based on MCG averages).
GMLOS_MAP = {
    '291': 4.1, '292': 3.5, '293': 2.8, # CHF
    '193': 4.2, '194': 3.6, # PNA
    '470': 2.4, # TKA
    '329': 6.5, # Bowel Surgery
    # Default fallback
    'DEFAULT': 3.0
}

# C. THE GUIDELINES LIBRARY (The "Rules Engine")
# Each dictionary represents one row from your Guidelines Table.
GUIDELINES_LIBRARY = [
    {
        "id": "CHF-001",
        "name": "Probable 'Wet' Discharge (Premature)",
        "logic": {
            "index_category": ["CHF"],
            "readm_category": ["CHF"],
            "max_days_gap": 7,
            "check_premature": True # Logic: Index LOS < GMLOS
        },
        "insight_template": "High-Risk Premature Discharge. Patient discharged {diff} days early. Audit Idea: Check final 24hr vitals, I/Os, and look for IV Lasix use."
    },
    {
        "id": "CHF-002",
        "name": "Failed Handoff (CHF)",
        "logic": {
            "index_category": ["CHF"],
            "disposition": ["01"], # 01 = Home
            "max_days_gap": 3
        },
        "insight_template": "Critical Handoff Failure. Patient bounced back in {gap} days after discharge to Home. Audit Idea: Order CM notes. Check for failed 'teach-back' on diet/meds and lack of scheduled follow-up."
    },
    {
        "id": "CHF-003",
        "name": "Renal Complication (CHF)",
        "logic": {
            "index_category": ["CHF"],
            "readm_category": ["RENAL_FAILURE"],
            "max_days_gap": 7
        },
        "insight_template": "Foreseeable Complication. Aggressive diuresis likely caused AKI. Audit Idea: Review index labs. Was Creatinine rising before discharge?"
    },
    {
        "id": "SURG-001",
        "name": "Surgical Site Infection (SSI)",
        "logic": {
            "index_category": ["TKA_THA", "BOWEL_SURGERY"],
            "readm_category": ["SEPSIS", "PJI"],
            "max_days_gap": 30
        },
        "insight_template": "Surgical Site Infection / Sepsis. Direct post-op complication. Audit Idea: Check discharge wound instructions and post-op vitals for ignored low-grade fevers."
    },
    {
        "id": "HANDOFF-001",
        "name": "SNF Bounce-Back",
        "logic": {
            "disposition": ["03"], # 03 = SNF
            "max_days_gap": 2
        },
        "insight_template": "Critical SNF Rejection. SNF likely rejected patient as unstable. Audit Idea: Compare transfer summary vitals vs. actual vitals. Look for documentation conflict in SNF readmit note."
    }
]

# =============================================================================
# 2. THE PROCESSING ENGINE (The Logic)
# =============================================================================

class ReadmissionAuditEngine:
    def __init__(self, drg_map, gmlos_map, guidelines):
        self.drg_map = drg_map
        self.gmlos_map = gmlos_map
        self.guidelines = guidelines

    def enrich_claim(self, claim):
        """
        Step 1: Enrich the raw claim with categories and benchmarks.
        """
        claim['index_category'] = self.drg_map.get(claim['index_drg'], 'OTHER')
        claim['readm_category'] = self.drg_map.get(claim['readm_drg'], 'OTHER')
        claim['gmlos'] = self.gmlos_map.get(claim['index_drg'], self.gmlos_map['DEFAULT'])
        
        # Calculate if discharge was early (for the 'check_premature' logic)
        claim['is_premature'] = claim['index_los'] < claim['gmlos']
        claim['los_diff'] = round(claim['gmlos'] - claim['index_los'], 1)
        
        return claim

    def evaluate_claim(self, raw_claim):
        """
        Step 2: Run the enriched claim against the Guidelines Library.
        """
        claim = self.enrich_claim(raw_claim.copy())
        triggered_insights = []

        for rule in self.guidelines:
            logic = rule['logic']
            is_match = True

            # --- LOGIC CHECKS ---
            
            # 1. Check Index DRG Category
            if 'index_category' in logic:
                if claim['index_category'] not in logic['index_category']:
                    is_match = False

            # 2. Check Readmission DRG Category
            if is_match and 'readm_category' in logic:
                if claim['readm_category'] not in logic['readm_category']:
                    is_match = False

            # 3. Check Days Gap (Time window)
            if is_match and 'max_days_gap' in logic:
                if claim['days_gap'] > logic['max_days_gap']:
                    is_match = False

            # 4. Check Disposition Code
            if is_match and 'disposition' in logic:
                if claim['disposition'] not in logic['disposition']:
                    is_match = False
            
            # 5. Check Premature Discharge Logic (LOS < GMLOS)
            if is_match and logic.get('check_premature'):
                if not claim['is_premature']:
                    is_match = False

            # --- RESULT GENERATION ---
            if is_match:
                # Format the insight string with dynamic data from the claim
                formatted_insight = rule['insight_template'].format(
                    diff=claim['los_diff'],
                    gap=claim['days_gap']
                )
                
                triggered_insights.append({
                    "Rule_ID": rule['id'],
                    "Rule_Name": rule['name'],
                    "Actionable_Insight": formatted_insight
                })

        claim['Audit_Results'] = triggered_insights
        claim['High_Priority'] = len(triggered_insights) > 0
        return claim

# =============================================================================
# 3. EXECUTION (Sample Run)
# =============================================================================

# Initialize the engine
engine = ReadmissionAuditEngine(DRG_CATEGORY_MAP, GMLOS_MAP, GUIDELINES_LIBRARY)

# Sample Claims Data (The Input)
sample_claims = [
    # Claim A: CHF patient, short stay, back in 3 days with CHF (Should trigger CHF-001 and CHF-002)
    {
        "claim_id": "CLM-001 (CHF High Risk)",
        "index_drg": "291", # CHF
        "readm_drg": "292", # CHF
        "index_los": 2.0,   # Short stay (GMLOS is 4.1)
        "days_gap": 3,
        "disposition": "01" # Home
    },
    # Claim B: TKA patient, back in 15 days with Infection (Should trigger SURG-001)
    {
        "claim_id": "CLM-002 (Surgical Infection)",
        "index_drg": "470", # Knee Replacement
        "readm_drg": "544", # PJI (Infection)
        "index_los": 3.0,
        "days_gap": 15,
        "disposition": "06" # Home Health
    },
    # Claim C: SNF Bounce back (Should trigger HANDOFF-001)
    {
        "claim_id": "CLM-003 (SNF Failure)",
        "index_drg": "194", # PNA
        "readm_drg": "871", # Sepsis
        "index_los": 4.0,
        "days_gap": 1,      # Back next day
        "disposition": "03" # SNF
    },
    # Claim D: Unrelated Trauma (Should trigger NOTHING)
    {
        "claim_id": "CLM-004 (Unrelated)",
        "index_drg": "291", # CHF
        "readm_drg": "999", # Trauma/Car Accident
        "index_los": 5.0,
        "days_gap": 12,
        "disposition": "01"
    }
]

# Run the claims through the engine
print(f"{'CLAIM ID':<30} | {'STATUS':<15} | {'RULES FIRED'}")
print("-" * 80)

for raw_claim in sample_claims:
    result = engine.evaluate_claim(raw_claim)
    
    status = "AUDIT" if result['High_Priority'] else "SKIP"
    rules_fired = [r['Rule_ID'] for r in result['Audit_Results']]
    
    print(f"{result['claim_id']:<30} | {status:<15} | {rules_fired}")
    
    # Print detailed insights for AUDIT claims
    if result['High_Priority']:
        for insight in result['Audit_Results']:
            print(f"   >> [{insight['Rule_ID']}] {insight['Actionable_Insight']}")
    print("-" * 80)
