import pandas as pd
import networkx as nx
from collections import defaultdict
import re
import time # For measuring execution time

# --- Configuration ---
MIN_TOTAL_CLAIMS_DEFAULT = 10 # Default minimum sample size for patterns
MIN_PROBABILITY_DEFAULT = 50 # Default minimum probability for patterns (in percent)

# --- 1. Create a Dummy DataFrame with more data for testing performance ---
# Increased data size significantly to simulate larger datasets and observe performance
num_claims = 10000 # Increased from 20 to 10000
data = {
    'claim_id': [f'C{i:05d}' for i in range(num_claims)],
    'diagnosis_code': ['I10-E11', 'J45', 'I10', 'E11-K21', 'J45', 'K21', 'I10-J45', 'J45', 'E11', 'I10'] * (num_claims // 10),
    'procedure_code': ['99213-71045', '99214', '99213', '99215-81000', '99214', '99213', '99213-74018', '99214', '99215', '99213'] * (num_claims // 10),
    'drg_code': [287, 204, 287, 637, 204, 287, 287, 204, 637, 287] * (num_claims // 10),
    'provider_id': [f'P{i % 10 + 1}' for i in range(num_claims)], # More providers
    'finding': ['Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'] * (num_claims // 10)
}

# Ensure data matches num_claims size (handle remainder if not perfectly divisible)
for col in data:
    data[col] = data[col][:num_claims]

df = pd.DataFrame(data)

print("Original DataFrame (simulated large data):")
print(f"Number of claims: {len(df)}")
# print(df.head()) # Don't print full df for large data
print("-" * 30)

# --- Helper function to split codes ---
def split_codes(code_string, delimiter='-'):
    """Splits a string of codes by a delimiter and returns a list of individual codes."""
    if isinstance(code_string, str):
        return [c.strip() for c in code_string.split(delimiter) if c.strip()]
    return []

# --- 2. Initialize Knowledge Graph ---
G = nx.DiGraph()

# --- Data structures for pre-calculation ---
# These will store counts directly during graph population
pre_problematic_signatures = defaultdict(int)
pre_total_signature_counts = defaultdict(int)

# --- 3. Populate the Knowledge Graph (with pre-calculation) ---
start_time = time.time()

# Define node prefixes for clarity
NODE_PREFIXES = {
    'claim': 'Claim_',
    'diag': 'Diagnosis_',
    'proc': 'Procedure_',
    'drg': 'DRG_',
    'provider': 'Provider_',
    'finding': 'Finding_'
}

# Identify the 'Yes' finding node string once
yes_finding_node_str = f"{NODE_PREFIXES['finding']}Yes"

for index, row in df.iterrows():
    claim_node = f"{NODE_PREFIXES['claim']}{row['claim_id']}"
    drg_node = f"{NODE_PREFIXES['drg']}{row['drg_code']}"
    provider_node = f"{NODE_PREFIXES['provider']}{row['provider_id']}"
    finding_node = f"{NODE_PREFIXES['finding']}{row['finding']}"

    G.add_node(claim_node, type='Claim', claim_id=row['claim_id'])
    G.add_node(drg_node, type='DRGCode', code=row['drg_code'])
    G.add_node(provider_node, type='Provider', id=row['provider_id'])
    G.add_node(finding_node, type='FindingStatus', status=row['finding'])

    G.add_edge(claim_node, drg_node, relation='ASSIGNED_DRG')
    G.add_edge(claim_node, provider_node, relation='BILLED_BY')
    G.add_edge(claim_node, finding_node, relation='HAS_FINDING_STATUS')

    individual_diag_codes = split_codes(row['diagnosis_code'])
    for diag_code in individual_diag_codes:
        diag_node = f"{NODE_PREFIXES['diag']}{diag_code}"
        G.add_node(diag_node, type='DiagnosisCode', code=diag_code)
        G.add_edge(claim_node, diag_node, relation='HAS_DIAGNOSIS')

    individual_proc_codes = split_codes(row['procedure_code'])
    for proc_code in individual_proc_codes:
        proc_node = f"{NODE_PREFIXES['proc']}{proc_code}"
        G.add_node(proc_node, type='ProcedureCode', code=proc_code)
        G.add_edge(claim_node, proc_node, relation='INCLUDES_PROCEDURE')

    original_diag_str = str(row['diagnosis_code'])
    original_proc_str = str(row['procedure_code'])
    combo_label = f"{original_diag_str}_{original_proc_str}_DRG_{row['drg_code']}"
    combo_node = f"Combo_{combo_label}"
    G.add_node(combo_node, type='CodeCombination',
               raw_diag_codes=original_diag_str,
               raw_proc_codes=original_proc_str,
               drg=row['drg_code'])
    G.add_edge(claim_node, combo_node, relation='REPRESENTS_CLAIM_SIGNATURE')
    G.add_edge(combo_node, finding_node, relation='LEADS_TO_FINDING_STATUS', claim_id=row['claim_id'])

    # --- Pre-calculation for problematic signatures ---
    # We are directly updating the counts here, avoiding a separate graph traversal later
    pre_total_signature_counts[combo_node] += 1
    if row['finding'] == 'Yes':
        pre_problematic_signatures[combo_node] += 1

end_time = time.time()
print(f"\nKnowledge Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges in {end_time - start_time:.2f} seconds.")
print("-" * 30)

# --- 4. Core Analysis Function (Now uses pre-calculated counts) ---

def analyze_problematic_signatures_optimized(graph, finding_node_prefix,
                                             pre_problematic_counts, pre_total_counts,
                                             min_total_claims_overall):
    """
    Analyzes the graph to find all claim signatures associated with 'Yes' findings,
    using pre-calculated counts and applying a minimum sample size threshold.
    Returns a sorted list of (signature_string, details_dict).
    """
    yes_finding_node = f"{finding_node_prefix}Yes"
    if not graph.has_node(yes_finding_node):
        # This case should ideally not happen if 'Yes' findings exist in data
        print(f"'{yes_finding_node}' node not found in the graph. Cannot analyze problematic signatures.")
        return []

    signature_analysis = {}
    for combo_node_id, total_count in pre_total_counts.items():
        # Apply the overall minimum sample size threshold here
        if total_count >= min_total_claims_overall:
            yes_count = pre_problematic_counts.get(combo_node_id, 0)
            percentage_yes = (yes_count / total_count) * 100
            signature_details = graph.nodes[combo_node_id] # Still need to get node properties from G

            original_signature_str = (
                f"{signature_details.get('raw_diag_codes', 'N/A')}_"
                f"{signature_details.get('raw_proc_codes', 'N/A')}_DRG_{signature_details.get('drg', 'N/A')}"
            )
            signature_analysis[original_signature_str] = {
                'yes_findings': yes_count,
                'total_claims': total_count,
                'percentage_yes': f"{percentage_yes:.2f}%"
            }

    sorted_signatures = sorted(signature_analysis.items(), key=lambda item: (float(item[1]['percentage_yes'].strip('%')), item[1]['total_claims']), reverse=True)
    # Sort by percentage_yes (desc), then total_claims (desc) for patterns with same percentage
    return sorted_signatures

# Run the initial analysis once to get all sorted problematic signatures
# Now uses the pre-calculated dictionaries and the default min_total_claims
start_analysis_time = time.time()
all_sorted_signatures = analyze_problematic_signatures_optimized(
    G,
    NODE_PREFIXES['finding'],
    pre_problematic_signatures,
    pre_total_signature_counts,
    min_total_claims_overall=MIN_TOTAL_CLAIMS_DEFAULT # Use the global default
)
end_analysis_time = time.time()
print(f"Initial pattern analysis completed in {end_analysis_time - start_analysis_time:.2f} seconds.")

if not all_sorted_signatures:
    print(f"No problematic signatures found in the initial analysis meeting min_total_claims={MIN_TOTAL_CLAIMS_DEFAULT}. Interactive mode will have no results.")

# --- 5. Interactive Query Function ---

def get_high_probability_patterns_for_drg(drg_input, all_sorted_signatures,
                                           probability_threshold=MIN_PROBABILITY_DEFAULT,
                                           min_total_claims=MIN_TOTAL_CLAIMS_DEFAULT):
    """
    Filters high probability patterns for a given DRG code, applying both
    a probability threshold and a minimum sample size threshold.

    Args:
        drg_input (str or int): The DRG code to filter by.
        all_sorted_signatures (list): The pre-computed list of all sorted problematic signatures.
        probability_threshold (int): The minimum percentage of 'Yes' findings.
        min_total_claims (int): The minimum number of occurrences (sample size).

    Returns:
        list: A list of (signature_string, details_dict) for the given DRG that meet the thresholds.
    """
    drg_str_match = f"DRG_{drg_input}"
    filtered_patterns = []

    print(f"\nSearching for high probability patterns for DRG: {drg_input} (Prob. Threshold: {probability_threshold}%, Sample Size: {min_total_claims})")

    for signature_str, details in all_sorted_signatures:
        # Check if the signature string contains the DRG code
        # AND meets the probability threshold
        # AND meets the minimum total claims threshold
        if (drg_str_match in signature_str and
            float(details['percentage_yes'].strip('%')) >= probability_threshold and
            details['total_claims'] >= min_total_claims):
            filtered_patterns.append((signature_str, details))

    return filtered_patterns

# --- 6. Interactive User Interface ---

print("\n--- Interactive DRG Pattern Finder ---")
print("Enter a DRG code to find high probability overpayment patterns.")
print(f"Default thresholds: Minimum Claims = {MIN_TOTAL_CLAIMS_DEFAULT}, Minimum Probability = {MIN_PROBABILITY_DEFAULT}%")
print("You can specify custom thresholds.")
print("Format: <DRG_CODE> [MIN_CLAIMS] [MIN_PROBABILITY_PERCENT]")
print(f"Examples: '287', '204 15', '637 {MIN_TOTAL_CLAIMS_DEFAULT} 75'")
print("Type 'all' to see all high probability patterns, or 'exit' to quit.")

while True:
    user_input_raw = input("\nEnter query (e.g., '287', '204 15', 'all', 'exit'): ").strip().upper()
    parts = user_input_raw.split()

    if parts[0] == 'EXIT':
        print("Exiting interactive mode. Goodbye!")
        break
    elif parts[0] == 'ALL':
        print(f"\n--- All High Probability Overpayment Patterns (Min Claims >= {MIN_TOTAL_CLAIMS_DEFAULT}, Prob. >= {MIN_PROBABILITY_DEFAULT}%) ---")
        found_any = False
        # 'all' command now uses the global default thresholds
        for signature_str, details in all_sorted_signatures:
            if (float(details['percentage_yes'].strip('%')) >= MIN_PROBABILITY_DEFAULT and
                details['total_claims'] >= MIN_TOTAL_CLAIMS_DEFAULT):
                print(f"- Signature: {signature_str}, Yes Findings: {details['yes_findings']}, Total Claims: {details['total_claims']}, % Yes: {details['percentage_yes']}")
                found_any = True
        if not found_any:
            print("No high probability patterns found overall meeting the default thresholds.")
    else:
        drg_code_query = None
        min_claims_query = MIN_TOTAL_CLAIMS_DEFAULT     # Default from config
        min_prob_query = MIN_PROBABILITY_DEFAULT      # Default from config

        try:
            drg_code_query = int(parts[0])
            if len(parts) > 1:
                min_claims_query = int(parts[1])
            if len(parts) > 2:
                min_prob_query = int(parts[2])
                if not (0 <= min_prob_query <= 100):
                    raise ValueError("Probability must be between 0 and 100.")
        except ValueError as e:
            print(f"Invalid input: {e}. Please use format '<DRG> [MIN_CLAIMS] [MIN_PROBABILITY]' or 'all' or 'exit'.")
            continue

        relevant_patterns = get_high_probability_patterns_for_drg(drg_code_query, all_sorted_signatures,
                                                                    probability_threshold=min_prob_query,
                                                                    min_total_claims=min_claims_query)

        if relevant_patterns:
            print(f"\n--- High Probability Overpayment Patterns for DRG {drg_code_query} (Sample Size >= {min_claims_query}, Prob. >= {min_prob_query}%) ---")
            for signature_str, details in relevant_patterns:
                print(f"- Signature: {signature_str}, Yes Findings: {details['yes_findings']}, Total Claims: {details['total_claims']}, % Yes: {details['percentage_yes']}")
        else:
            print(f"No high probability overpayment patterns found for DRG {drg_code_query} meeting the specified thresholds (Sample Size >= {min_claims_query}, Prob. >= {min_prob_query}%).")

