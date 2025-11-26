import pandas as pd
import numpy as np

class APR_DRG_Audit_Engine:
    def __init__(self, df):
        self.raw_df = df
        self.stats = None

    def preprocess_and_train(self):
        """
        1. Calculates the 'Severity Profile' (Count of Sev 1, 2, 3, 4) for each claim.
        2. Flattens the data.
        3. Learns drop patterns based on the specific mix of severities.
        """
        print("--- Analyzing Claim Composition (Severity Profiling) ---")
        
        flattened_data = []
        
        for idx, row in self.raw_df.iterrows():
            claim_id = row['ClaimID']
            apr_drg = row['APR_DRG_Code']
            claim_sev = row['Claim_SOI'] # The final SOI (1-4)
            
            diag_dict = row['Diag_Codes_Dict'] # {Code: Severity_Int}
            dropped_set = set(row['Dropped_Codes']) if isinstance(row['Dropped_Codes'], list) else set()
            
            # --- INTELLIGENCE STEP: Calculate Severity Counts ---
            # We count how many codes of each severity exist on this specific claim
            # This detects if a claim is "Fragile" (only 1 high severity code) or "Robust"
            sev_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            
            for code, sev in diag_dict.items():
                # Ensure sev is an integer (handle potential string inputs)
                sev_int = int(sev)
                if sev_int in sev_counts:
                    sev_counts[sev_int] += 1
            
            # --- EXPLOSION STEP ---
            for code, code_sev in diag_dict.items():
                code_sev_int = int(code_sev)
                
                # We identify if this specific code is a "Lone Driver"
                # (i.e., It is a Sev 4 code, and there is only 1 Sev 4 code on the claim)
                is_lone_driver = 1 if (code_sev_int == 4 and sev_counts[4] == 1) else 0
                
                flattened_data.append({
                    'APR_DRG': apr_drg,
                    'Claim_SOI': claim_sev,
                    'Diag_Code': code,
                    'Code_SOI': code_sev_int,
                    # We Embed the Profile into the row
                    'Count_SOI_3': sev_counts[3],
                    'Count_SOI_4': sev_counts[4],
                    'Is_Lone_Driver': is_lone_driver, 
                    'Is_Dropped': 1 if code in dropped_set else 0
                })
        
        long_df = pd.DataFrame(flattened_data)
        
        print("--- Learning Logic based on Severity Counts ---")
        
        # --- TRAINING STEP ---
        # We group by the Counts of High Severity codes.
        # This allows us to learn: "J96.00 is dropped when there are NO other Sev 4 codes"
        self.stats = long_df.groupby([
            'APR_DRG', 
            'Claim_SOI', 
            'Diag_Code', 
            'Count_SOI_3', 
            'Count_SOI_4'
        ]).agg(
            Total_Appearances=('Is_Dropped', 'count'),
            Drop_Count=('Is_Dropped', 'sum')
        ).reset_index()
        
        self.stats['Drop_Probability'] = self.stats['Drop_Count'] / self.stats['Total_Appearances']
        
        # Filter for noise (require at least 2 examples to form a rule)
        self.stats = self.stats[self.stats['Total_Appearances'] >= 2]
        
        return self.stats

    def generate_smart_recommendations(self, threshold=0.4):
        print(f"--- Generating Smart Recommendations (Confidence > {threshold*100}%) ---")
        
        high_risk = self.stats[self.stats['Drop_Probability'] >= threshold].sort_values(
            by='Drop_Probability', ascending=False
        )
        
        recs = []
        for _, row in high_risk.iterrows():
            # We construct a sentence that explains the logic based on the COUNTS
            
            # Context description
            context = []
            if row['Count_SOI_4'] > 0:
                context.append(f"{row['Count_SOI_4']}x SOI-4 codes")
            if row['Count_SOI_3'] > 0:
                context.append(f"{row['Count_SOI_3']}x SOI-3 codes")
            
            context_str = " and ".join(context) if context else "low severity codes"
            
            sentence = (
                f"For APR-DRG {row['APR_DRG']} (SOI {row['Claim_SOI']}): "
                f"When the claim contains [{context_str}], "
                f"Code {row['Diag_Code']} is DROPPED. "
                f"(Confidence: {int(row['Drop_Probability']*100)}%)"
            )
            recs.append(sentence)
            
        return recs

# ==========================================
# 1. SIMULATE APR-DRG DATA
# ==========================================

# Scenario 1: The "Fragile" Claim. 
# J96.00 is the ONLY Severity 4 code. Auditors love to drop this to lower payment.
claim_fragile = {
    'ClaimID': 1, 'APR_DRG_Code': 137, 'Claim_SOI': 4,
    'Diag_Codes_Dict': {'J18.9': 2, 'I10': 1, 'J96.00': 4}, # Only one 4
    'Dropped_Codes': ['J96.00']
}

# Scenario 2: The "Robust" Claim.
# Claim has multiple Sev 4 codes. Dropping J96.00 doesn't change much.
claim_robust = {
    'ClaimID': 2, 'APR_DRG_Code': 137, 'Claim_SOI': 4,
    'Diag_Codes_Dict': {'J18.9': 2, 'I50.23': 4, 'N17.9': 4, 'J96.00': 4}, # Three 4s
    'Dropped_Codes': [] # Not dropped because the claim stays Sev 4 anyway
}

# Add more data to reinforce the pattern
data = [
    claim_fragile, 
    claim_robust,
    # Another fragile case reinforcing the rule
    {'ClaimID': 3, 'APR_DRG_Code': 137, 'Claim_SOI': 4, 
     'Diag_Codes_Dict': {'E11.9': 2, 'J96.00': 4}, 'Dropped_Codes': ['J96.00']}, 
    # Another robust case reinforcing the rule
    {'ClaimID': 4, 'APR_DRG_Code': 137, 'Claim_SOI': 4, 
     'Diag_Codes_Dict': {'I50.23': 4, 'J96.00': 4}, 'Dropped_Codes': []}
]

df = pd.DataFrame(data)

# ==========================================
# 2. RUN ENGINE
# ==========================================

engine = APR_DRG_Audit_Engine(df)
engine.preprocess_and_train()
recommendations = engine.generate_smart_recommendations()

print("\n--- OUTPUT ---")
for r in recommendations:
    print(r)
