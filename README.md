import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import time
import re

# --- Configuration ---
# These thresholds are crucial for ARM and will heavily influence the output.
# Adjust based on your data volume and desired specificity/generality of rules.
MIN_SUPPORT_PERCENT = 0.01 # Minimum support as a fraction (e.g., 0.01 = 1% of claims)
MIN_CONFIDENCE = 0.50      # Minimum confidence (e.g., 0.50 = 50% confident)
MIN_LIFT = 1.0             # Minimum lift (typically > 1 for positive association)

# For interactive query
DEFAULT_QUERY_MIN_SUPPORT_PERCENT = 0.01
DEFAULT_QUERY_MIN_CONFIDENCE = 0.60 # Slightly higher default for query for stricter rules

# --- 1. Create a Dummy DataFrame with more data ---
num_claims = 10000
data = {
    'claim_id': [f'C{i:05d}' for i in range(num_claims)],
    'diagnosis_code': ['I10-E11', 'J45', 'I10', 'E11-K21', 'J45', 'K21', 'I10-J45', 'J45', 'E11', 'I10'] * (num_claims // 10),
    'procedure_code': ['99213-71045', '99214', '99213', '99215-81000', '99214', '99213', '99213-74018', '99214', '99215', '99213'] * (num_claims // 10),
    'drg_code': [287, 204, 287, 637, 204, 287, 287, 204, 637, 287] * (num_claims // 10),
    'provider_id': [f'P{i % 10 + 1}' for i in range(num_claims)],
    'finding': ['Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'] * (num_claims // 10)
}
for col in data:
    data[col] = data[col][:num_claims]
df = pd.DataFrame(data)

print("Original DataFrame (simulated large data):")
print(f"Number of claims: {len(df)}")
print("-" * 30)

# --- Helper function to split codes ---
def split_codes(code_string, delimiter='-'):
    """Splits a string of codes by a delimiter and returns a list of individual codes."""
    if isinstance(code_string, str):
        return [c.strip() for c in code_string.split(delimiter) if c.strip()]
    return []

# --- 2. Data Transformation for ARM (One-Hot Encoding) ---
start_time = time.time()

# Create a list of lists, where each inner list represents the items in a claim
# We'll prefix items to differentiate them (e.g., 'diag:I10', 'proc:99213', 'drg:287', 'finding:Yes')
transactions = []
all_unique_items = set()

for index, row in df.iterrows():
    current_transaction_items = []

    # Add diagnosis codes
    for diag_code in split_codes(row['diagnosis_code']):
        item = f"diag:{diag_code}"
        current_transaction_items.append(item)
        all_unique_items.add(item)

    # Add procedure codes
    for proc_code in split_codes(row['procedure_code']):
        item = f"proc:{proc_code}"
        current_transaction_items.append(item)
        all_unique_items.add(item)

    # Add DRG code
    item = f"drg:{row['drg_code']}"
    current_transaction_items.append(item)
    all_unique_items.add(item)

    # Add Finding status
    item = f"finding:{row['finding']}"
    current_transaction_items.append(item)
    all_unique_items.add(item)

    transactions.append(current_transaction_items)

# Create a one-hot encoded DataFrame
oht = pd.DataFrame(transactions)
oht = oht.stack().str.get_dummies().sum(level=0) # Sum for claims with multiple codes in one field
oht = oht.astype(bool) # Convert to boolean for apriori efficiency

end_time = time.time()
print(f"Data transformation for ARM completed in {end_time - start_time:.2f} seconds.")
print(f"One-Hot Encoded DataFrame shape: {oht.shape}")
print(f"First 5 rows of OHT DataFrame:\n{oht.head()}")
print("-" * 30)

# --- 3. Apply Apriori Algorithm to find frequent itemsets ---
start_apriori_time = time.time()

# min_support is a fraction of total transactions
frequent_itemsets = apriori(oht, min_support=MIN_SUPPORT_PERCENT, use_colnames=True)

end_apriori_time = time.time()
print(f"Apriori algorithm completed in {end_apriori_time - start_apriori_time:.2f} seconds.")
print(f"Found {len(frequent_itemsets)} frequent itemsets (min_support={MIN_SUPPORT_PERCENT}).")
# print(frequent_itemsets.head())
print("-" * 30)

# --- 4. Generate Association Rules ---
start_rules_time = time.time()

# Generate rules based on confidence and lift
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
rules = rules[rules['lift'] >= MIN_LIFT] # Filter by lift as well

end_rules_time = time.time()
print(f"Association rules generation completed in {end_rules_time - start_rules_time:.2f} seconds.")
print(f"Found {len(rules)} association rules (min_confidence={MIN_CONFIDENCE}, min_lift={MIN_LIFT}).")
# print(rules.head())
print("-" * 30)

# --- 5. Filter for Overpayment Rules and Pre-process for faster querying ---
start_filter_time = time.time()

overpayment_rules = rules[rules['consequents'].apply(lambda x: 'finding:Yes' in x)].copy()
overpayment_rules = overpayment_rules.sort_values(by=['confidence', 'support'], ascending=False)

# Add a DRG column to the rules for easy filtering later
overpayment_rules['antecedent_drg'] = overpayment_rules['antecedents'].apply(
    lambda x: next((item.split(':')[1] for item in x if item.startswith('drg:')), None)
)

end_filter_time = time.time()
print(f"Filtered {len(overpayment_rules)} overpayment rules in {end_filter_time - start_filter_time:.2f} seconds.")
# print(overpayment_rules.head())
print("-" * 30)

# --- 6. Interactive Query Function ---

def query_overpayment_rules_by_drg(drg_code, all_rules_df, min_support_pct, min_confidence_val):
    """
    Queries the pre-computed overpayment rules for a specific DRG code,
    applying minimum support and confidence thresholds.

    Args:
        drg_code (str or int): The DRG code to filter by.
        all_rules_df (pd.DataFrame): The DataFrame of all pre-computed overpayment rules.
        min_support_pct (float): Minimum support as a percentage (e.g., 0.01 for 1%).
        min_confidence_val (float): Minimum confidence (e.g., 0.50 for 50%).

    Returns:
        pd.DataFrame: Filtered rules.
    """
    target_drg_str = str(drg_code)
    
    # Filter by DRG, then by support and confidence
    filtered_rules = all_rules_df[
        (all_rules_df['antecedent_drg'] == target_drg_str) &
        (all_rules_df['support'] >= min_support_pct) &
        (all_rules_df['confidence'] >= min_confidence_val)
    ].copy()

    # Format antecedents for display
    filtered_rules['antecedents_formatted'] = filtered_rules['antecedents'].apply(
        lambda x: ', '.join(sorted([item for item in x if item != f'drg:{target_drg_str}']))
    )
    
    return filtered_rules.sort_values(by=['confidence', 'support'], ascending=False)

# --- 7. Interactive User Interface ---

print("\n--- Interactive Overpayment Rule Finder (Association Rule Mining) ---")
print(f"Default thresholds: Min Support = {DEFAULT_QUERY_MIN_SUPPORT_PERCENT*100:.2f}%, Min Confidence = {DEFAULT_QUERY_MIN_CONFIDENCE*100:.2f}%")
print("Enter DRG code to find overpayment patterns (rules).")
print("You can also specify custom thresholds for support and confidence.")
print("Format: <DRG_CODE> [MIN_SUPPORT_PERCENT] [MIN_CONFIDENCE]")
print("Examples: '287', '204 0.02', '637 0.015 0.70'")
print("Type 'all' to see all overpayment rules, or 'exit' to quit.")

while True:
    user_input_raw = input("\nEnter query (e.g., '287', '204 0.02', 'all', 'exit'): ").strip().upper()
    parts = user_input_raw.split()

    if parts[0] == 'EXIT':
        print("Exiting interactive mode. Goodbye!")
        break
    elif parts[0] == 'ALL':
        print(f"\n--- All Overpayment Rules (Min Support >= {DEFAULT_QUERY_MIN_SUPPORT_PERCENT*100:.2f}%, Min Confidence >= {DEFAULT_QUERY_MIN_CONFIDENCE*100:.2f}%) ---")
        
        # Filter all rules using default query thresholds
        all_display_rules = overpayment_rules[
            (overpayment_rules['support'] >= DEFAULT_QUERY_MIN_SUPPORT_PERCENT) &
            (overpayment_rules['confidence'] >= DEFAULT_QUERY_MIN_CONFIDENCE)
        ].copy()
        
        if not all_display_rules.empty:
            all_display_rules['antecedents_formatted'] = all_display_rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
            for _, rule in all_display_rules.iterrows():
                print(f"Rule: {rule['antecedents_formatted']} => {list(rule['consequents'])[0]}")
                print(f"  Support: {rule['support']:.4f} ({rule['support']*100:.2f}%), Confidence: {rule['confidence']:.4f} ({rule['confidence']*100:.2f}%), Lift: {rule['lift']:.2f}")
        else:
            print("No overpayment rules found overall meeting the default thresholds.")
    else:
        drg_code_query = None
        min_support_query = DEFAULT_QUERY_MIN_SUPPORT_PERCENT
        min_confidence_query = DEFAULT_QUERY_MIN_CONFIDENCE

        try:
            drg_code_query = int(parts[0])
            if len(parts) > 1:
                min_support_query = float(parts[1])
                if not (0 <= min_support_query <= 1):
                    raise ValueError("Support must be a fraction between 0 and 1.")
            if len(parts) > 2:
                min_confidence_query = float(parts[2])
                if not (0 <= min_confidence_query <= 1):
                    raise ValueError("Confidence must be a fraction between 0 and 1.")
        except ValueError as e:
            print(f"Invalid input: {e}. Please use format '<DRG> [MIN_SUPPORT] [MIN_CONFIDENCE]' or 'all' or 'exit'.")
            continue

        relevant_rules = query_overpayment_rules_by_drg(
            drg_code_query,
            overpayment_rules, # Use the pre-filtered overpayment rules
            min_support_query,
            min_confidence_query
        )

        if not relevant_rules.empty:
            print(f"\n--- Overpayment Rules for DRG {drg_code_query} (Min Support >= {min_support_query*100:.2f}%, Min Confidence >= {min_confidence_query*100:.2f}%) ---")
            for _, rule in relevant_rules.iterrows():
                print(f"Rule: {{{rule['antecedents_formatted']}}} => {{finding:Yes}}")
                print(f"  Support: {rule['support']:.4f} ({rule['support']*100:.2f}%), Confidence: {rule['confidence']:.4f} ({rule['confidence']*100:.2f}%), Lift: {rule['lift']:.2f}")
        else:
            print(f"No overpayment rules found for DRG {drg_code_query} meeting the specified thresholds.")

