import pandas as pd
import networkx as nx
from collections import defaultdict

# --- 1. Create a Dummy DataFrame (Simulate your data) ---
# Your DataFrame should ideally have a unique claim ID for each row.
data = {
    'claim_id': ['C001', 'C002', 'C003', 'C004', 'C005', 'C006', 'C007', 'C008', 'C009', 'C010', 'C011', 'C012', 'C013'],
    'diagnosis_code': ['I10', 'J45', 'I10', 'E11', 'J45', 'K21', 'I10', 'J45', 'E11', 'I10', 'I10', 'J45', 'I10'],
    'procedure_code': ['99213', '99214', '99213', '99215', '99214', '99213', '99213', '99214', '99215', '99213', '99213', '99214', '99213'],
    'drg_code': [287, 204, 287, 637, 204, 287, 287, 204, 637, 287, 287, 204, 287],
    'provider_id': ['P1', 'P2', 'P1', 'P3', 'P2', 'P4', 'P1', 'P2', 'P3', 'P1', 'P5', 'P2', 'P1'],
    'finding': ['Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'] # 'Yes' for overpayment/finding
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
print("-" * 30)

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
    # Create unique node IDs for each entity type
    claim_node = f"{NODE_PREFIXES['claim']}{row['claim_id']}"
    diag_node = f"{NODE_PREFIXES['diag']}{row['diagnosis_code']}"
    proc_node = f"{NODE_PREFIXES['proc']}{row['procedure_code']}"
    drg_node = f"{NODE_PREFIXES['drg']}{row['drg_code']}"
    provider_node = f"{NODE_PREFIXES['provider']}{row['provider_id']}"
    finding_node = f"{NODE_PREFIXES['finding']}{row['finding']}"

    # Add nodes with their types (properties)
    G.add_node(claim_node, type='Claim', claim_id=row['claim_id'])
    G.add_node(diag_node, type='DiagnosisCode', code=row['diagnosis_code'])
    G.add_node(proc_node, type='ProcedureCode', code=row['procedure_code'])
    G.add_node(drg_node, type='DRGCode', code=row['drg_code'])
    G.add_node(provider_node, type='Provider', id=row['provider_id'])
    G.add_node(finding_node, type='FindingStatus', status=row['finding'])

    # Add relationships (edges)
    G.add_edge(claim_node, diag_node, relation='HAS_PRIMARY_DIAGNOSIS')
    G.add_edge(claim_node, proc_node, relation='INCLUDES_PROCEDURE')
    G.add_edge(claim_node, drg_node, relation='ASSIGNED_DRG')
    G.add_edge(claim_node, provider_node, relation='BILLED_BY')
    G.add_edge(claim_node, finding_node, relation='HAS_FINDING_STATUS')

    # Add direct co-occurrence edges between codes for easier querying
    combo_label = f"{row['diagnosis_code']}_{row['procedure_code']}_DRG_{row['drg_code']}"
    combo_node = f"Combo_{combo_label}"
    G.add_node(combo_node, type='CodeCombination', diag=row['diagnosis_code'], proc=row['procedure_code'], drg=row['drg_code'])
    G.add_edge(claim_node, combo_node, relation='REPRESENTS_COMBO')
    G.add_edge(combo_node, finding_node, relation='LEADS_TO_FINDING_STATUS', claim_id=row['claim_id']) # Link combo to finding

print(f"\nKnowledge Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
print("Example nodes and edges:")
# Print a few nodes and edges to verify
nodes_to_show = list(G.nodes())[:5]
edges_to_show = list(G.edges())[:5]
print("Nodes:", nodes_to_show, "...")
print("Edges:", edges_to_show, "...")
print("-" * 30)

# --- 4. Querying the Knowledge Graph for Relationships ---

print("\n--- KG Query Results for Overpayment Identification ---")

# Goal 1: Find which (Diagnosis, Procedure, DRG) combinations are most frequently associated with a 'Yes' finding.

yes_finding_node = f"{NODE_PREFIXES['finding']}Yes"
if G.has_node(yes_finding_node):
    problematic_combos = defaultdict(int) # Stores counts of how many times a combo led to 'Yes'
    total_combo_counts = defaultdict(int) # Stores total counts for each combo

    # Iterate over all edges to find the 'LEADS_TO_FINDING_STATUS' edges
    # Correct unpacking: source_node, target_node, edge_attributes_dict
    for source_node, target_node, edge_attributes in G.edges(data=True):
        # Line 90 was here in the previous version, now updated:
        if edge_attributes.get('relation') == 'LEADS_TO_FINDING_STATUS':
            # Check if this edge points to the 'Yes' finding node
            if target_node == yes_finding_node:
                # The source_node here is the 'Combo_' node
                problematic_combos[source_node] += 1

            # Count total occurrences of this combo node (regardless of finding status)
            # This logic assumes each edge from Combo_node to Finding_node represents one claim
            total_combo_counts[source_node] += 1

    print("\nCombinations most associated with 'Yes' Findings:")
    combo_analysis = {}
    for combo_node_id, yes_count in problematic_combos.items():
        total_count = total_combo_counts.get(combo_node_id, 0)
        if total_count > 0:
            percentage_yes = (yes_count / total_count) * 100
            combo_details = G.nodes[combo_node_id]
            original_combo_str = f"{combo_details['diag']}_{combo_details['proc']}_DRG_{combo_details['drg']}"
            combo_analysis[original_combo_str] = {
                'yes_findings': yes_count,
                'total_claims': total_count,
                'percentage_yes': f"{percentage_yes:.2f}%"
            }

    # Sort and display results
    sorted_combos = sorted(combo_analysis.items(), key=lambda item: float(item[1]['percentage_yes'].strip('%')), reverse=True)
    for combo_str, details in sorted_combos:
        print(f"- Combo: {combo_str}, Yes Findings: {details['yes_findings']}, Total Claims: {details['total_claims']}, % Yes: {details['percentage_yes']}")

    if not sorted_combos:
        print("No 'Yes' findings found or no combinations linked to them.")

else:
    print(f"'{yes_finding_node}' node not found in the graph.")


# Goal 2: Identify providers who frequently bill these problematic combinations.
print("\n--- Providers Associated with Problematic Combinations ---")
problematic_providers_counts = defaultdict(int)

# Filter for combos with a high percentage of 'Yes' findings (e.g., > 50%)
high_risk_combos = [combo_str for combo_str, details in sorted_combos if float(details['percentage_yes'].strip('%')) > 50]

for claim_node in G.nodes():
    if G.nodes[claim_node].get('type') == 'Claim':
        current_diag = None
        current_proc = None
        current_drg = None
        current_provider_node = None
        has_yes_finding = False

        # Get relevant neighbors and their attributes
        for neighbor_node in G.successors(claim_node):
            if G.nodes[neighbor_node].get('type') == 'DiagnosisCode':
                current_diag = G.nodes[neighbor_node].get('code')
            elif G.nodes[neighbor_node].get('type') == 'ProcedureCode':
                current_proc = G.nodes[neighbor_node].get('code')
            elif G.nodes[neighbor_node].get('type') == 'DRGCode':
                current_drg = G.nodes[neighbor_node].get('code')
            elif G.nodes[neighbor_node].get('type') == 'Provider':
                current_provider_node = neighbor_node
            elif G.nodes[neighbor_node].get('type') == 'FindingStatus' and G.nodes[neighbor_node].get('status') == 'Yes':
                has_yes_finding = True

        # Check if this claim matches a high-risk combo AND has a 'Yes' finding
        if current_diag and current_proc and current_drg and current_provider_node and has_yes_finding:
            claim_combo_str = f"{current_diag}_{current_proc}_DRG_{current_drg}"
            if claim_combo_str in high_risk_combos:
                problematic_providers_counts[current_provider_node] += 1


# Display problematic providers
if problematic_providers_counts:
    sorted_providers = sorted(problematic_providers_counts.items(), key=lambda item: item[1], reverse=True)
    for provider_node_id, count in sorted_providers:
        provider_id = G.nodes[provider_node_id]['id']
        print(f"- Provider {provider_id} (Node: {provider_node_id}) associated with {count} problematic 'Yes' claims.")
else:
    print("No providers found frequently associated with problematic 'Yes' claims.")

