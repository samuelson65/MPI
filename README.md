import pandas as pd
import networkx as nx
from collections import defaultdict
import re

# --- 1. Create a Dummy DataFrame with multiple codes per column ---
# Note: Using '-' as the delimiter for demonstration. Adjust 'split_codes' if needed.
data = {
    'claim_id': ['C001', 'C002', 'C003', 'C004', 'C005', 'C006', 'C007', 'C008', 'C009', 'C010', 'C011', 'C012', 'C013'],
    'diagnosis_code': ['I10-E11', 'J45', 'I10', 'E11-K21', 'J45', 'K21', 'I10-J45', 'J45', 'E11', 'I10', 'I10', 'J45', 'I10'],
    'procedure_code': ['99213-71045', '99214', '99213', '99215-81000', '99214', '99213', '99213-74018', '99214', '99215', '99213', '99213', '99214', '99213'],
    'drg_code': [287, 204, 287, 637, 204, 287, 287, 204, 637, 287, 287, 204, 287],
    'provider_id': ['P1', 'P2', 'P1', 'P3', 'P2', 'P4', 'P1', 'P2', 'P3', 'P1', 'P5', 'P2', 'P1'],
    'finding': ['Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'] # 'Yes' for overpayment/finding
}
df = pd.DataFrame(data)

print("Original DataFrame (with multi-codes):")
print(df)
print("-" * 30)

# --- Helper function to split codes ---
def split_codes(code_string, delimiter='-'):
    """Splits a string of codes by a delimiter and returns a list of individual codes."""
    if isinstance(code_string, str):
        return [c.strip() for c in code_string.split(delimiter) if c.strip()]
    return [] # Return empty list if not a string (e.g., NaN)

# --- 2. Initialize Knowledge Graph ---
G = nx.DiGraph()

# --- 3. Populate the Knowledge Graph ---

# Define node prefixes for clarity
NODE_PREFIXES = {
    'claim': 'Claim_',
    'diag': 'Diagnosis_',
    'proc': 'Procedure_',
    'drg': 'DRG_',
    'provider': 'Provider_',
    'finding': 'Finding_'
}

for index, row in df.iterrows():
    # Create unique node IDs for main entities
    claim_node = f"{NODE_PREFIXES['claim']}{row['claim_id']}"
    drg_node = f"{NODE_PREFIXES['drg']}{row['drg_code']}"
    provider_node = f"{NODE_PREFIXES['provider']}{row['provider_id']}"
    finding_node = f"{NODE_PREFIXES['finding']}{row['finding']}"

    # Add main nodes with properties
    G.add_node(claim_node, type='Claim', claim_id=row['claim_id'])
    G.add_node(drg_node, type='DRGCode', code=row['drg_code'])
    G.add_node(provider_node, type='Provider', id=row['provider_id'])
    G.add_node(finding_node, type='FindingStatus', status=row['finding'])

    # Add relationships for main entities
    G.add_edge(claim_node, drg_node, relation='ASSIGNED_DRG')
    G.add_edge(claim_node, provider_node, relation='BILLED_BY')
    G.add_edge(claim_node, finding_node, relation='HAS_FINDING_STATUS')

    # Handle multiple diagnosis codes
    individual_diag_codes = split_codes(row['diagnosis_code'])
    for diag_code in individual_diag_codes:
        diag_node = f"{NODE_PREFIXES['diag']}{diag_code}"
        G.add_node(diag_node, type='DiagnosisCode', code=diag_code)
        G.add_edge(claim_node, diag_node, relation='HAS_DIAGNOSIS')

    # Handle multiple procedure codes
    individual_proc_codes = split_codes(row['procedure_code'])
    for proc_code in individual_proc_codes:
        proc_node = f"{NODE_PREFIXES['proc']}{proc_code}"
        G.add_node(proc_node, type='ProcedureCode', code=proc_code)
        G.add_edge(claim_node, proc_node, relation='INCLUDES_PROCEDURE')

    # Create the CodeCombination node based on the original concatenated strings
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

print(f"\nKnowledge Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
print("Example nodes and edges (first 10):")
nodes_to_show = list(G.nodes())[:10]
edges_to_show = list(G.edges())[:10]
print("Nodes:", nodes_to_show, "...")
print("Edges:", edges_to_show, "...")
print("-" * 30)

# --- 4. Core Analysis Function (Non-interactive part) ---

def analyze_problematic_signatures(graph, finding_node_prefix):
    """
    Analyzes the graph to find all claim signatures associated with 'Yes' findings.
    Returns a sorted list of (signature_string, details_dict).
    """
    yes_finding_node = f"{finding_node_prefix}Yes"
    if not graph.has_node(yes_finding_node):
        print(f"'{yes_finding_node}' node not found in the graph. Cannot analyze problematic signatures.")
        return []

    problematic_signatures = defaultdict(int)
    total_signature_counts = defaultdict(int)

    for source_node, target_node, edge_attributes in graph.edges(data=True):
        if edge_attributes.get('relation') == 'LEADS_TO_FINDING_STATUS':
            if target_node == yes_finding_node:
                problematic_signatures[source_node] += 1
            total_signature_counts[source_node] += 1

    signature_analysis = {}
    for combo_node_id, yes_count in problematic_signatures.items():
        total_count = total_signature_counts.get(combo_node_id, 0)
        if total_count > 0:
            percentage_yes = (yes_count / total_count) * 100
            signature_details = graph.nodes[combo_node_id]
            original_signature_str = (
                f"{signature_details.get('raw_diag_codes', 'N/A')}_"
                f"{signature_details.get('raw_proc_codes', 'N/A')}_DRG_{signature_details.get('drg', 'N/A')}"
            )
            signature_analysis[original_signature_str] = {
                'yes_findings': yes_count,
                'total_claims': total_count,
                'percentage_yes': f"{percentage_yes:.2f}%"
            }

    sorted_signatures = sorted(signature_analysis.items(), key=lambda item: float(item[1]['percentage_yes'].strip('%')), reverse=True)
    return sorted_signatures

# Run the initial analysis once to get all sorted problematic signatures
all_sorted_signatures = analyze_problematic_signatures(G, NODE_PREFIXES['finding'])

if not all_sorted_signatures:
    print("No problematic signatures found in the initial analysis. Interactive mode will have no results.")

# --- 5. Interactive Query Function ---

def get_high_probability_patterns_for_drg(drg_input, all_sorted_signatures, probability_threshold=50):
    """
    Filters high probability patterns for a given DRG code.

    Args:
        drg_input (str or int): The DRG code to filter by.
        all_sorted_signatures (list): The pre-computed list of all sorted problematic signatures.
        probability_threshold (int): The minimum percentage of 'Yes' findings to consider a pattern high probability.

    Returns:
        list: A list of (signature_string, details_dict) for the given DRG that meet the threshold.
    """
    drg_str_match = f"DRG_{drg_input}"
    filtered_patterns = []

    print(f"\nSearching for high probability patterns for DRG: {drg_input} (Threshold: {probability_threshold}%)")

    for signature_str, details in all_sorted_signatures:
        # Check if the signature string contains the DRG code and meets the probability threshold
        if drg_str_match in signature_str and float(details['percentage_yes'].strip('%')) >= probability_threshold:
            filtered_patterns.append((signature_str, details))

    return filtered_patterns

# --- 6. Interactive User Interface ---

print("\n--- Interactive DRG Pattern Finder ---")
print("Enter a DRG code to find high probability overpayment patterns.")
print("Type 'all' to see all high probability patterns, or 'exit' to quit.")

while True:
    user_input = input("\nEnter DRG code (or 'all', 'exit'): ").strip().upper()

    if user_input == 'EXIT':
        print("Exiting interactive mode. Goodbye!")
        break
    elif user_input == 'ALL':
        print("\n--- All High Probability Overpayment Patterns (Threshold: 50%) ---")
        found_any = False
        for signature_str, details in all_sorted_signatures:
            if float(details['percentage_yes'].strip('%')) >= 50: # Default threshold for 'all'
                print(f"- Signature: {signature_str}, Yes Findings: {details['yes_findings']}, Total Claims: {details['total_claims']}, % Yes: {details['percentage_yes']}")
                found_any = True
        if not found_any:
            print("No high probability patterns found overall.")
    else:
        # Assume user input is a DRG code
        try:
            drg_code_query = int(user_input) # Convert to int if it's a number
        except ValueError:
            print("Invalid input. Please enter a numerical DRG code, 'all', or 'exit'.")
            continue

        relevant_patterns = get_high_probability_patterns_for_drg(drg_code_query, all_sorted_signatures)

        if relevant_patterns:
            print(f"\n--- High Probability Overpayment Patterns for DRG {drg_code_query} ---")
            for signature_str, details in relevant_patterns:
                print(f"- Signature: {signature_str}, Yes Findings: {details['yes_findings']}, Total Claims: {details['total_claims']}, % Yes: {details['percentage_yes']}")
        else:
            print(f"No high probability overpayment patterns found for DRG {drg_code_query} meeting the threshold (50%).")

# Optional: You can add other interactive queries here, e.g.,
# "Find providers associated with a specific pattern"
# "Find individual codes common in a specific DRG's problematic patterns"
