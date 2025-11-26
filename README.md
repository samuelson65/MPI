import pandas as pd
import numpy as np

class AuditRecommender:
    def __init__(self, df):
        self.raw_df = df
        self.stats = None
        self.rules = None

    def preprocess_and_train(self):
        """
        1. Explodes the Dictionary Column into multiple rows.
        2. Flags which codes were dropped.
        3. Calculates Drop Probability for every Code/DRG/Severity combo.
        """
        print("--- Processing Data ---")
        
        # --- STEP 1: FLATTEN THE DICTIONARY (SCALABILITY KEY) ---
        # We use a list comprehension which is faster than apply() for this specific structure
        flattened_data = []
        
        for idx, row in self.raw_df.iterrows():
            claim_id = row['ClaimID']
            drg = row['DRG_Code']
            claim_sev = row['Claim_Severity']
            dropped_set = set(row['Dropped_Codes']) if isinstance(row['Dropped_Codes'], list) else set()
            
            # Iterate through the dictionary of codes for this specific claim
            # Input format: {'J96.00': 'MCC', 'I10': 'Non-CC'}
            if isinstance(row['Diag_Codes_Dict'], dict):
                for code, code_severity in row['Diag_Codes_Dict'].items():
                    flattened_data.append({
                        'ClaimID': claim_id,
                        'DRG': drg,
                        'Claim_Severity': claim_sev,
                        'Diag_Code': code,
                        'Code_Severity_Type': code_severity, # e.g., MCC/CC
                        'Is_Dropped': 1 if code in dropped_set else 0
                    })
        
        long_df = pd.DataFrame(flattened_data)
        
        # --- STEP 2: AGGREGATE (TRAINING) ---
        # We group by the Claim Context to find patterns
        print("--- Learning Patterns ---")
        
        self.stats = long_df.groupby(['DRG', 'Claim_Severity', 'Diag_Code']).agg(
            Total_Appearances=('Is_Dropped', 'count'),
            Drop_Count=('Is_Dropped', 'sum')
        ).reset_index()
        
        # Calculate Probability
        self.stats['Drop_Probability'] = self.stats['Drop_Count'] / self.stats['Total_Appearances']
        
        # Filter out noise (e.g., codes that appeared less than 3 times)
        self.stats = self.stats[self.stats['Total_Appearances'] >= 3]
        
        return self.stats

    def generate_recommendations(self, threshold=0.3):
        """
        Generates the specific 'Given DRG...' sentences.
        """
        print(f"--- Generating Rules (Confidence > {threshold*100}%) ---")
        
        # Sort by highest probability of being dropped
        prioritized_rules = self.stats[self.stats['Drop_Probability'] >= threshold].sort_values(
            by=['Drop_Probability', 'Total_Appearances'], ascending=False
        )
        
        recommendations = []
        
        for _, row in prioritized_rules.iterrows():
            # Format the sentence exactly as requested
            sentence = (
                f"Given DRG code {row['DRG']} and Severity {row['Claim_Severity']}, "
                f"the code {row['Diag_Code']} can be DROPPED "
                f"(Historical Drop Rate: {round(row['Drop_Probability']*100, 1)}%)"
            )
            recommendations.append(sentence)
            
        return recommendations

# ==========================================
# 1. INPUT DATA SIMULATION
# ==========================================
# This matches the structure you described:
# - Diag_Codes_Dict: Key=Code, Value=Severity (MCC/CC)
# - Dropped_Codes: List of codes removed
data = [
    {
        'ClaimID': 101, 
        'DRG_Code': 871, 
        'Claim_Severity': 3, 
        'Diag_Codes_Dict': {'A41.9': 'MCC', 'J96.00': 'MCC', 'I10': 'Non-CC'}, 
        'Dropped_Codes': ['J96.00'] # J96 dropped
    },
    {
        'ClaimID': 102, 
        'DRG_Code': 871, 
        'Claim_Severity': 3, 
        'Diag_Codes_Dict': {'A41.9': 'MCC', 'J96.00': 'MCC', 'E11.9': 'CC'}, 
        'Dropped_Codes': ['J96.00'] # J96 dropped again (Pattern!)
    },
    {
        'ClaimID': 103, 
        'DRG_Code': 190, 
        'Claim_Severity': 1, 
        'Diag_Codes_Dict': {'J18.9': 'CC', 'I10': 'Non-CC'}, 
        'Dropped_Codes': [] # Nothing dropped
    },
    {
        'ClaimID': 104, 
        'DRG_Code': 871, 
        'Claim_Severity': 3, 
        'Diag_Codes_Dict': {'A41.9': 'MCC', 'J96.00': 'MCC'}, 
        'Dropped_Codes': [] # J96 kept this time
    },
     {
        'ClaimID': 105, 
        'DRG_Code': 871, 
        'Claim_Severity': 3, 
        'Diag_Codes_Dict': {'A41.9': 'MCC', 'J96.00': 'MCC', 'N17.9': 'CC'}, 
        'Dropped_Codes': ['J96.00'] # J96 dropped 3rd time
    }
]

df = pd.DataFrame(data)

# ==========================================
# 2. RUN THE ENGINE
# ==========================================

engine = AuditRecommender(df)
engine.preprocess_and_train()
results = engine.generate_recommendations(threshold=0.5) # 50% confidence threshold

# ==========================================
# 3. OUTPUT
# ==========================================
print("\nOUTPUT:")
for res in results:
    print(res)
