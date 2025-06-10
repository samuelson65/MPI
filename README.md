import pandas as pd
import networkx as nx
from collections import defaultdict
import re # For splitting codes using regex if delimiters vary

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
        G.add_edge(claim_node, diag_node, relation='HAS_DIAGNOSIS') # Use HAS_DIAGNOSIS for all

    # Handle multiple procedure codes
    individual_proc_codes = split_codes(row['procedure_code'])
    for proc_code in individual_proc_codes:
        proc_node = f"{NODE_PREFIXES['proc']}{proc_code}"
        G.add_node(proc_node, type='ProcedureCode', code=proc_code)
        G.add_edge(claim_node, proc_node, relation='INCLUDES_PROCEDURE')


    # Create the CodeCombination node based on the original concatenated strings
    # This represents the specific "claim signature" that occurred.
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
print("Example nodes and edges:")
# Print a few nodes and edges to verify
nodes_to_show = list(G.nodes())[:10] # Show more to see individual codes
edges_to_show = list(G.edges())[:10]
print("Nodes:", nodes_to_show, "...")
print("Edges:", edges_to_show, "...")
print("-" * 30)

# --- 4. Querying the Knowledge Graph for Relationships ---

print("\n--- KG Query Results for Overpayment Identification ---")

# Goal 1: Find which (Diagnosis, Procedure, DRG) "claim signatures" are most frequently associated with a 'Yes' finding.

yes_finding_node = f"{NODE_PREFIXES['finding']}Yes"
if G.has_node(yes_finding_node):
    problematic_signatures = defaultdict(int) # Stores counts of how many times a signature led to 'Yes'
    total_signature_counts = defaultdict(int) # Stores total counts for each signature

    for source_node, target_node, edge_attributes in G.edges(data=True):
        if edge_attributes.get('relation') == 'LEADS_TO_FINDING_STATUS':
            if target_node == yes_finding_node:
                problematic_signatures[source_node] += 1 # source_node is the 'Combo_' node
            total_signature_counts[source_node] += 1

    print("\nClaim Signatures most associated with 'Yes' Findings:")
    signature_analysis = {}
    for combo_node_id, yes_count in problematic_signatures.items():
        total_count = total_signature_counts.get(combo_node_id, 0)
        if total_count > 0:
            percentage_yes = (yes_count / total_count) * 100
            signature_details = G.nodes[combo_node_id]
            # Use the original raw strings for the signature for clarity
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
    for signature_str, details in sorted_signatures:
        print(f"- Signature: {signature_str}, Yes Findings: {details['yes_findings']}, Total Claims: {details['total_claims']}, % Yes: {details['percentage_yes']}")

    if not sorted_signatures:
        print("No 'Yes' findings found or no signatures linked to them.")

else:
    print(f"'{yes_finding_node}' node not found in the graph.")


# Goal 2: Identify providers who frequently bill these problematic claim signatures.
print("\n--- Providers Associated with Problematic Claim Signatures ---")
problematic_providers_counts = defaultdict(int)

# Filter for signatures with a high percentage of 'Yes' findings (e.g., > 50%)
high_risk_signatures = {combo_str for combo_str, details in sorted_signatures if float(details['percentage_yes'].strip('%')) > 50}

for claim_node in G.nodes():
    if G.nodes[claim_node].get('type') == 'Claim':
        # Find the claim's associated provider and its signature status
        current_provider_node = None
        has_yes_finding = False
        current_signature_combo_node = None

        for neighbor_node in G.successors(claim_node):
            if G.nodes[neighbor_node].get('type') == 'Provider':
                current_provider_node = neighbor_node
            elif G.nodes[neighbor_node].get('type') == 'FindingStatus' and G.nodes[neighbor_node].get('status') == 'Yes':
                has_yes_finding = True
            elif G.nodes[neighbor_node].get('type') == 'CodeCombination':
                current_signature_combo_node = neighbor_node

        if current_provider_node and has_yes_finding and current_signature_combo_node:
            signature_details = G.nodes[current_signature_combo_node]
            claim_signature_str = (
                f"{signature_details.get('raw_diag_codes', 'N/A')}_"
                f"{signature_details.get('raw_proc_codes', 'N/A')}_DRG_{signature_details.get('drg', 'N/A')}"
            )

            if claim_signature_str in high_risk_signatures:
                problematic_providers_counts[current_provider_node] += 1


# Display problematic providers
if problematic_providers_counts:
    sorted_providers = sorted(problematic_providers_counts.items(), key=lambda item: item[1], reverse=True)
    for provider_node_id, count in sorted_providers:
        provider_id = G.nodes[provider_node_id]['id']
        print(f"- Provider {provider_id} (Node: {provider_node_id}) associated with {count} problematic 'Yes' claims.")
else:
    print("No providers found frequently associated with problematic 'Yes' claims.")


# --- New Goal: Explore relationships between individual codes within problematic signatures ---
print("\n--- Exploring Individual Code Relationships in Problematic Signatures ---")

# Let's find specific individual diagnosis and procedure codes that frequently appear
# within the high-risk claim signatures.

# Collect individual problematic diagnosis and procedure codes
problematic_individual_diag_codes = defaultdict(int)
problematic_individual_proc_codes = defaultdict(int)

for signature_str in high_risk_signatures:
    # Parse the original raw string back to its components
    parts = signature_str.split('_DRG_')
    raw_diag_part = parts[0].split('_')[0]
    raw_proc_part = parts[0].split('_')[1]

    # Split the raw parts into individual codes
    individual_diags = split_codes(raw_diag_part)
    individual_procs = split_codes(raw_proc_part)

    for diag_code in individual_diags:
        problematic_individual_diag_codes[diag_code] += 1
    for proc_code in individual_procs:
        problematic_individual_proc_codes[proc_code] += 1

print("\nTop Individual Diagnosis Codes in High-Risk Signatures:")
sorted_diags = sorted(problematic_individual_diag_codes.items(), key=lambda item: item[1], reverse=True)
for diag, count in sorted_diags:
    print(f"- {diag}: Appears in {count} high-risk signatures.")

print("\nTop Individual Procedure Codes in High-Risk Signatures:")
sorted_procs = sorted(problematic_individual_proc_codes.items(), key=lambda item: item[1], reverse=True)
for proc, count in sorted_procs:
    print(f"- {proc}: Appears in {count} high-risk signatures.")

print("\n--- Next Steps for Deeper Insight ---")
print("Now you have:")
print("1. Specific **claim signatures** (full raw diagnosis-procedure-DRG strings) most correlated with 'Yes' findings.")
print("2. The **providers** frequently associated with these signatures.")
print("3. The **individual diagnosis and procedure codes** that are commonly part of these high-risk signatures.")
print("\nFor further analysis, you could:")
print("- Investigate claims where a 'Yes' finding occurred but the signature is NOT in the high-risk list (potential new patterns).")
print("- Analyze the specific combination of *individual* diagnosis and procedure codes that appear together within high-risk claims (e.g., using graph algorithms like finding common neighbors or subgraph isomorphism if your graph is large enough).")
print("- If using a proper graph database (like Neo4j), write complex Cypher queries to find paths like 'Provider X -> Claim -> (Individual Diag A & Individual Proc B) -> DRG C -> Finding Yes'.")

