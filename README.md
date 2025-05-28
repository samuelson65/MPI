import pandas as pd

# --- 1. Initialize the Knowledge Graph Data Structure ---
kg_nodes = {}  # {node_id: {'labels': [], 'properties': {}}}
kg_edges = []  # [(source_id, relationship_type, target_id)]

# Helper to add a node and ensure uniqueness by ID
def add_node(node_id, labels, properties=None):
    if node_id not in kg_nodes:
        kg_nodes[node_id] = {'labels': labels if isinstance(labels, list) else [labels], 'properties': properties if properties else {}}
    else:
        # If node exists, merge labels and properties
        existing_labels = kg_nodes[node_id]['labels']
        for label in (labels if isinstance(labels, list) else [labels]):
            if label not in existing_labels:
                existing_labels.append(label)
        if properties:
            kg_nodes[node_id]['properties'].update(properties)
    return node_id

# Helper to add an edge
def add_edge(source_id, relationship_type, target_id):
    kg_edges.append((source_id, relationship_type, target_id))

# --- 2. Sample Data Ingestion ---

# a) Sample Claims Data
claims_data = [
    {"claim_id": "C1001", "patient_id": "P001", "drg_code": "DRG-100", "los": 15,
     "principal_diagnosis": "I10", "secondary_diagnoses": ["E11.9"],
     "procedures": ["Z98.89"], "provider_id": "PRV001", "discharge_disposition": "Home"},
    {"claim_id": "C1002", "patient_id": "P002", "drg_code": "DRG-100", "los": 5, # Suspiciously low LOS
     "principal_diagnosis": "I10", "secondary_diagnoses": [],
     "procedures": ["Z98.89"], "provider_id": "PRV001", "discharge_disposition": "Home"},
    {"claim_id": "C1003", "patient_id": "P003", "drg_code": "DRG-200", "los": 7,
     "principal_diagnosis": "J18.9", "secondary_diagnoses": [],
     "procedures": [], "provider_id": "PRV002", "discharge_disposition": "SNF"},
    {"claim_id": "C1004", "patient_id": "P004", "drg_code": "DRG-100", "los": 30, # High LOS
     "principal_diagnosis": "I10", "secondary_diagnoses": ["J45.9"], # Asthma as secondary
     "procedures": ["Z98.89"], "provider_id": "PRV001", "discharge_disposition": "Home"}
]

# b) Sample DRG Typical Values
drg_definitions = {
    "DRG-100": {"typical_los_min": 10, "typical_los_max": 20, "typical_procedures": ["Z98.89"]},
    "DRG-200": {"typical_los_min": 5, "typical_los_max": 12, "typical_procedures": []}
}

# c) Sample SME and Policy Data
sme_data = {"SME001": "Dr. Sarah Lee"}
policy_data = {"POLICY001": "CMS Chapter 3 - Inpatient Payment"}

# d) Pre-existing SME-defined concepts/rules
sme_rules = [
    {"rule_id": "RULE_LOS_DRG100_Short", "drg_code": "DRG-100", "los_threshold": 8,
     "description": "DRG-100 with LOS less than 8 typically indicates overpayment unless specific complications.",
     "defined_by": "SME001", "based_on": "POLICY001"}
]

print("Ingesting data into the conceptual Knowledge Graph...")

# Ingest DRG nodes and their typical values
for drg_code, props in drg_definitions.items():
    add_node(f"DRG_{drg_code}", "DRG", properties={
        "code": drg_code,
        "typical_los_min": props["typical_los_min"],
        "typical_los_max": props["typical_los_max"]
    })
    for proc_code in props["typical_procedures"]:
        add_node(f"PROC_{proc_code}", "Procedure", {"code": proc_code})
        add_edge(f"DRG_{drg_code}", "TYPICALLY_INCLUDES_PROCEDURE", f"PROC_{proc_code}")

# Ingest Claims and related entities
for claim_dict in claims_data:
    claim_id = f"CLAIM_{claim_dict['claim_id']}"
    patient_id = f"PATIENT_{claim_dict['patient_id']}"
    drg_id = f"DRG_{claim_dict['drg_code']}"
    provider_id = f"PROVIDER_{claim_dict['provider_id']}"
    discharge_disposition_id = f"DISP_{claim_dict['discharge_disposition'].replace(' ', '_')}"
    principal_diag_id = f"DIAG_{claim_dict['principal_diagnosis']}"

    # Add nodes
    add_node(claim_id, "Claim", {"id": claim_dict["claim_id"], "los": claim_dict["los"]})
    add_node(patient_id, "Patient", {"id": claim_dict["patient_id"]})
    add_node(drg_id, "DRG", {"code": claim_dict["drg_code"]}) # Ensures DRG node exists even if not in definitions
    add_node(provider_id, "Provider", {"id": claim_dict["provider_id"]})
    add_node(discharge_disposition_id, "DischargeDisposition", {"type": claim_dict["discharge_disposition"]})
    add_node(principal_diag_id, "Diagnosis", {"code": claim_dict["principal_diagnosis"]})

    # Add relationships for principal claim data
    add_edge(claim_id, "HAS_PATIENT", patient_id)
    add_edge(claim_id, "ASSIGNED_TO_DRG", drg_id)
    add_edge(claim_id, "BILLED_BY", provider_id)
    add_edge(claim_id, "HAS_DISCHARGE_DISPOSITION", discharge_disposition_id)
    add_edge(claim_id, "HAS_PRINCIPAL_DIAGNOSIS", principal_diag_id)

    # Add secondary diagnoses
    for sec_diag_code in claim_dict.get("secondary_diagnoses", []):
        sec_diag_id = f"DIAG_{sec_diag_code}"
        add_node(sec_diag_id, "Diagnosis", {"code": sec_diag_code})
        add_edge(claim_id, "HAS_SECONDARY_DIAGNOSIS", sec_diag_id)

    # Add procedures
    for proc_code in claim_dict.get("procedures", []):
        proc_id = f"PROC_{proc_code}"
        add_node(proc_id, "Procedure", {"code": proc_code})
        add_edge(claim_id, "HAS_PROCEDURE", proc_id)

# Ingest SME and Policy nodes and rules
for sme_id, sme_name in sme_data.items():
    add_node(f"SME_{sme_id}", "SME", {"id": sme_id, "name": sme_name})

for policy_id, policy_name in policy_data.items():
    add_node(f"POLICY_{policy_id}", "Policy", {"id": policy_id, "name": policy_name})

for rule_dict in sme_rules:
    rule_id = f"RULE_{rule_dict['rule_id']}"
    drg_id = f"DRG_{rule_dict['drg_code']}"
    sme_id = f"SME_{rule_dict['defined_by']}"
    policy_id = f"POLICY_{rule_dict['based_on']}"

    add_node(rule_id, "Rule", {
        "id": rule_dict["rule_id"],
        "description": rule_dict["description"],
        "los_threshold": rule_dict["los_threshold"]
    })
    add_edge(rule_id, "APPLIES_TO_DRG", drg_id)
    add_edge(rule_id, "DEFINED_BY_SME", sme_id)
    add_edge(rule_id, "IS_BASED_ON_POLICY", policy_id)

print("Data ingestion complete.")
print(f"Total nodes: {len(kg_nodes)}")
print(f"Total edges: {len(kg_edges)}")

# --- 3. Simulating Graph Queries ---

# Helper function to find connected nodes via a relationship type
def get_connected_nodes(source_id, relationship_type, target_label=None):
    results = []
    for s, r_type, t in kg_edges:
        if s == source_id and r_type == relationship_type:
            if target_label:
                if target_label in kg_nodes[t]['labels']:
                    results.append(kg_nodes[t])
            else:
                results.append(kg_nodes[t])
    return results

# Helper function to find sources connected to a target via a relationship type
def get_sources_connected_to(target_id, relationship_type, source_label=None):
    results = []
    for s, r_type, t in kg_edges:
        if t == target_id and r_type == relationship_type:
            if source_label:
                if source_label in kg_nodes[s]['labels']:
                    results.append(kg_nodes[s])
            else:
                results.append(kg_nodes[s])
    return results

print("\n--- Simulating Queries ---")

# --- Query 1: SME-driven concept (direct rule application) ---
# Find claims violating the SME-defined short LOS rule for DRG-100
print("\nQuery 1: Claims violating SME-defined rule (DRG-100, LOS < 8)")
# Find the rule node
rule_node_id = "RULE_RULE_LOS_DRG100_Short"
rule_props = kg_nodes.get(rule_node_id, {}).get('properties', {})
los_threshold = rule_props.get('los_threshold')
rule_description = rule_props.get('description')

matching_claims_q1 = []
if los_threshold is not None:
    # Find DRG-100 node
    drg100_node = None
    for node_id, node_data in kg_nodes.items():
        if 'DRG' in node_data['labels'] and node_data['properties'].get('code') == 'DRG-100':
            drg100_node = node_id
            break

    if drg100_node:
        # Find claims assigned to DRG-100
        for s_id, r_type, t_id in kg_edges:
            if r_type == "ASSIGNED_TO_DRG" and t_id == drg100_node and 'Claim' in kg_nodes[s_id]['labels']:
                claim_node = kg_nodes[s_id]
                if claim_node['properties'].get('los') < los_threshold:
                    matching_claims_q1.append({
                        "ClaimID": claim_node['properties']['id'],
                        "LOS": claim_node['properties']['los'],
                        "DRG": kg_nodes[drg100_node]['properties']['code'],
                        "RuleDescription": rule_description
                    })

if matching_claims_q1:
    print(pd.DataFrame(matching_claims_q1))
else:
    print("No claims found matching this rule.")

# --- Query 2: Data-driven concept discovery (finding anomalies based on typical values) ---
# Find DRG-100 claims where LOS is significantly outside the typical range,
# AND no secondary diagnoses or specific procedures are present.
print("\nQuery 2: Claims where DRG-100 LOS is unusually low (below typical_los_min) without secondary diagnoses/procedures")

matching_claims_q2 = []
drg100_node_id = None
for node_id, node_data in kg_nodes.items():
    if 'DRG' in node_data['labels'] and node_data['properties'].get('code') == 'DRG-100':
        drg100_node_id = node_id
        break

if drg100_node_id:
    drg100_props = kg_nodes[drg100_node_id]['properties']
    typical_los_min = drg100_props.get('typical_los_min')

    if typical_los_min is not None:
        for s_id, r_type, t_id in kg_edges:
            if r_type == "ASSIGNED_TO_DRG" and t_id == drg100_node_id and 'Claim' in kg_nodes[s_id]['labels']:
                claim_node = kg_nodes[s_id]
                claim_los = claim_node['properties'].get('los')

                if claim_los is not None and claim_los < typical_los_min:
                    # Check for secondary diagnoses
                    has_secondary_diag = False
                    for edge_s, edge_r, edge_t in kg_edges:
                        if edge_s == s_id and edge_r == "HAS_SECONDARY_DIAGNOSIS":
                            has_secondary_diag = True
                            break

                    # Check for procedures
                    has_procedure = False
                    for edge_s, edge_r, edge_t in kg_edges:
                        if edge_s == s_id and edge_r == "HAS_PROCEDURE":
                            has_procedure = True
                            break

                    if not has_secondary_diag and not has_procedure:
                        matching_claims_q2.append({
                            "ClaimID": claim_node['properties']['id'],
                            "LOS": claim_los,
                            "DRG": drg100_props['code'],
                            "TypicalLOSMin": typical_los_min
                        })

df_result_2 = pd.DataFrame(matching_claims_q2)
if not df_result_2.empty:
    print(df_result_2)
    print("\nPROPOSED NEW CONCEPT:")
    print("  'DRG-100 claims with LOS < Typical_Min_LOS (e.g., 10) AND no secondary diagnoses AND no procedures.'")
    print("  This pattern was detected from data anomaly relative to DRG definitions.")
else:
    print("No claims found matching this anomalous pattern.")

# --- Query 3: Explain why CatBoost might flag C1002 as overpayment (using KG details) ---
print("\nQuery 3: Explain why CatBoost might flag C1002 as overpayment (using KG details)")
target_claim_id = "CLAIM_C1002"
claim_details = {}

if target_claim_id in kg_nodes:
    claim_node = kg_nodes[target_claim_id]
    claim_details['ClaimID'] = claim_node['properties']['id']
    claim_details['LOS'] = claim_node['properties']['los']

    # Get DRG details
    connected_drgs = get_connected_nodes(target_claim_id, "ASSIGNED_TO_DRG", "DRG")
    if connected_drgs:
        drg_props = connected_drgs[0]['properties']
        claim_details['DRG'] = drg_props.get('code')
        claim_details['TypicalDRGLOSMin'] = drg_props.get('typical_los_min')

    # Get Principal Diagnosis
    principal_diag = get_connected_nodes(target_claim_id, "HAS_PRINCIPAL_DIAGNOSIS", "Diagnosis")
    if principal_diag:
        claim_details['PrincipalDiagnosis'] = principal_diag[0]['properties']['code']

    # Get Secondary Diagnoses
    secondary_diagnoses = get_connected_nodes(target_claim_id, "HAS_SECONDARY_DIAGNOSIS", "Diagnosis")
    claim_details['SecondaryDiagnoses'] = [d['properties']['code'] for d in secondary_diagnoses]

    # Get Procedures
    procedures = get_connected_nodes(target_claim_id, "HAS_PROCEDURE", "Procedure")
    claim_details['Procedures'] = [p['properties']['code'] for p in procedures]

    # Get Provider
    provider = get_connected_nodes(target_claim_id, "BILLED_BY", "Provider")
    if provider:
        claim_details['ProviderID'] = provider[0]['properties']['id']

    if claim_details:
        print(f"Claim {claim_details['ClaimID']} details:")
        print(f"  LOS: {claim_details.get('LOS')} (Typical DRG {claim_details.get('DRG')} LOS Min: {claim_details.get('TypicalDRGLOSMin')})")
        print(f"  Principal Diagnosis: {claim_details.get('PrincipalDiagnosis')}")
        print(f"  Secondary Diagnoses: {claim_details.get('SecondaryDiagnoses')}")
        print(f"  Procedures: {claim_details.get('Procedures')}")
        print(f"  Billed by Provider: {claim_details.get('ProviderID')}")

        # Inferential step (manual in this script, but could be automated)
        if claim_details.get('LOS', 0) < claim_details.get('TypicalDRGLOSMin', 0) and not claim_details['SecondaryDiagnoses']:
            print("\nKG Inference/Explanation: This claim's LOS is below the typical minimum for its DRG, and it lacks secondary diagnoses that might justify a longer stay or additional complexity, making it suspicious for overpayment due to short stay.")
    else:
        print("Claim C1002 found, but details incomplete.")
else:
    print("Claim C1002 not found in the KG.")


print("\nScript finished.")
