import pandas as pd
import numpy as np

class AuditRuleGenerator:
    def __init__(self, data):
        self.df = data
        self.stats = None
        self.rules = None

    def preprocess_wide_to_long(self):
        """
        Converts 25 Dx columns into a single vertical 'Code' column.
        This is the secret to scalability.
        """
        print("1. Reshaping data from Wide (25 cols) to Long (1 col)...")
        
        # 1. Identify Dx columns
        dx_cols = [col for col in self.df.columns if col.startswith('Original_Dx')]
        final_dx_cols = [col for col in self.df.columns if col.startswith('Final_Dx')]

        # 2. Melt Original Codes (Create one row per code)
        # We keep ClaimID, DRG, and Severity as identifiers
        df_long = self.df.melt(
            id_vars=['ClaimID', 'DRG', 'Severity'], 
            value_vars=dx_cols, 
            value_name='Code'
        ).dropna(subset=['Code']) # Remove empty slots
        
        # 3. Create a set of "Final Valid Codes" for fast lookup
        # We consolidate all final codes into a set per ClaimID for O(1) lookup
        print("2. Indexing Final Audited Codes...")
        
        # Helper to collect non-null final codes into a set
        def gather_codes(row):
            codes = [row[c] for c in final_dx_cols if pd.notna(row[c])]
            return set(codes)
            
        final_lookup = self.df.set_index('ClaimID').apply(gather_codes, axis=1)
        
        # 4. Map the check: Is the specific 'Code' in 'Final_Codes'?
        # If it is NOT in the final set, it was DROPPED.
        df_long['Is_Dropped'] = df_long.apply(
            lambda row: row['Code'] not in final_lookup.get(row['ClaimID'], set()), axis=1
        )
        
        return df_long

    def train_logic(self, df_long):
        """
        Aggregates millions of rows instantly using GroupBy
        """
        print("3. calculating probabilities...")
        
        # Group by the specific context: DRG + Severity + Code
        # We calculate count (Total) and sum (Dropped Count)
        self.stats = df_long.groupby(['DRG', 'Severity', 'Code'])['Is_Dropped'].agg(
            Total_Appearances='count',
            Drop_Count='sum'
        ).reset_index()
        
        # Calculate Probability
        self.stats['Drop_Probability'] = self.stats['Drop_Count'] / self.stats['Total_Appearances']
        
        # Filter noise (e.g., code only appeared once)
        self.stats = self.stats[self.stats['Total_Appearances'] >= 5]
        
        return self.stats

    def generate_sentences(self, confidence_threshold=0.4):
        """
        Converts the math back into English sentences
        """
        print("4. Generating Recommendations...")
        
        # Filter for high risk
        high_risk = self.stats[self.stats['Drop_Probability'] >= confidence_threshold].sort_values(
            by='Drop_Probability', ascending=False
        )
        
        recs = []
        for _, row in high_risk.iterrows():
            sentence = (
                f"Given DRG {int(row['DRG'])} and Severity {int(row['Severity'])}, "
                f"the code {row['Code']} can be DROPPED "
                f"(Historical Fail Rate: {round(row['Drop_Probability']*100, 1)}% "
                f"based on {row['Total_Appearances']} claims)"
            )
            recs.append(sentence)
        
        return recs

# ==========================================
# USAGE SIMULATION
# ==========================================

# 1. Create Dummy Data with 25 Columns (Real-world structure)
# Imagine 1000 claims
data_size = 1000
data = {
    'ClaimID': range(data_size),
    'DRG': np.random.choice([871, 190, 291, 194], size=data_size),
    'Severity': np.random.choice([1, 2, 3, 4], size=data_size),
}

# Simulate 25 Dx columns (Mostly empty/NaN for realism)
for i in range(1, 26):
    # Randomly assign codes, with some NaNs
    col_name = f"Original_Dx_{i}"
    data[col_name] = np.random.choice(['J96.00', 'I10', 'E11.9', 'A41.9', np.nan], size=data_size)

df = pd.DataFrame(data)

# Simulate "Final" columns (Where J96.00 is often removed if DRG=871)
# Copy original to final first
for i in range(1, 26):
    df[f"Final_Dx_{i}"] = df[f"Original_Dx_{i}"]

# Inject Logic: If DRG is 871 and Code is J96.00, drop it in Final columns
# (This simulates the audit action)
for idx, row in df.iterrows():
    if row['DRG'] == 871:
        for i in range(1, 26):
            if row[f"Original_Dx_{i}"] == 'J96.00':
                df.at[idx, f"Final_Dx_{i}"] = np.nan # Simulate Drop

# ==========================================
# EXECUTE ENGINE
# ==========================================

engine = AuditRuleGenerator(df)

# Step A: Preprocess
long_data = engine.preprocess_wide_to_long()

# Step B: Train
engine.train_logic(long_data)

# Step C: Get Results
recommendations = engine.generate_sentences(confidence_threshold=0.5)

print("\n--- AUDIT RECOMMENDATIONS (SCALED) ---")
for r in recommendations[:10]: # Print top 10
    print(r)
