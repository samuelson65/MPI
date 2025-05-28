import pandas as pd

# --- 1. Initialize the Knowledge Graph Data Structure ---
# Nodes are stored in a dictionary where keys are unique node IDs.
# Each node has 'labels' (e.g., 'Claim', 'DRG') and 'properties' (attributes).
kg_nodes = {}  # {node_id: {'labels': [], 'properties': {}}}

# Edges (relationships) are stored as a list of tuples: (source_id, relationship_type, target_id).
kg_edges = []  # [(source_id, relationship_type, target_id)]

# Helper to add a node to the KG, ensures uniqueness by ID and merges properties/labels if node exists
def add_node(node_id, labels, properties=None):
    if node_id not in kg_nodes:
        kg_nodes[node_id] = {'labels': labels if isinstance(labels, list) else [labels], 'properties': properties if properties else {}}
    else:
        # If node exists, merge new labels and properties
        existing_labels = kg_nodes[node_id]['labels']
        for label in (labels if isinstance(labels, list) else [labels]):
            if label not in existing_labels:
                existing_labels.append(label)
        if properties:
            kg_nodes[node_id]['properties'].update(properties)
    return node_id

# Helper to add an edge (relationship) to the KG
def add_edge(source_id, relationship_type, target_id):
    # Basic check to ensure source and target nodes exist before adding an edge
    if source_id in kg_nodes and target_id in kg_nodes:
        kg_edges.append((source_id, relationship_type, target_id))
    else:
        print(f"Warning: Attempted to add edge {source_id}-{relationship_type}->{target_id} but one or both nodes do not exist.")

# --- 2. Sample Data Ingestion ---

# Simulate your input dataset as a pandas DataFrame
# This includes provider_id, drg_code, comorbidities, discharge_status, Length_of_Stay,
# and other typical claim variables.
your_dataset = pd.DataFrame([
    {"claim_id": "C1001", "provider_id": "PRV001", "drg_code": "DRG-100", "comorbidities": "E11.9,I10", "discharge_status": "Home", "los": 15, "principal_diagnosis": "I10", "procedures": "Z98.89"},
    {"claim_id": "C1002", "provider_id": "PRV001", "drg_code": "DRG-100", "comorbidities": "", "discharge_status": "Home", "los": 5, "principal_diagnosis": "I10", "procedures": "Z98.89"}, # Suspiciously low LOS
    {"claim_id": "C1003", "provider_id": "PRV002", "drg_code": "DRG-200", "comorbidities": "J45.9", "discharge_status": "SNF", "los": 7, "principal_diagnosis": "J18.9", "procedures": ""},
    {"claim_id": "C1004", "provider_id": "PRV001", "drg_code": "DRG-100", "comorbidities": "I50.9,N18.9", "discharge_status": "Home", "los": 30, "principal_diagnosis": "I10", "procedures": "Z98.89"} # High LOS
])

# Additional knowledge for DRG typical values (from your SMEs or official sources)
drg_definitions = {
    "DRG-100": {
        "typical_los_min": 10, "typical_los_max": 20,
        "typical_procedures": ["Z98.89", "5A1945Z"], # Added another typical procedure
        "common_diagnoses": ["I10", "E11.9"], # Common for this DRG
        "major_complications": ["J45.9", "I50.9"] # These might extend LOS or justify higher payment
    },
    "DRG-200": {
        "typical_los_min": 5, "typical_los_max": 12,
        "typical_procedures": ["0B1G3FZ"],
        "common_diagnoses": ["J18.9"],
        "major_complications": []
    }
}

# SME-defined rules (like your conditional queries) - these become explicit "Rule" nodes in the KG
sme_rules = [
    {"rule_id": "RULE_DRG100_LOS_Short_NoComorbid", "drg_code": "DRG-100", "los_threshold": 8,
     "description": "DRG-100 with LOS < 8 and NO major comorbidities indicates overpayment.",
     "defined_by": "SME001", "triggers_overpayment_category": "Short_Stay_No_Comorbid"},
    {"rule_id": "RULE_DRG100_LOS_Long_NoJustify", "drg_code": "DRG-100", "los_threshold": 25,
     "description": "DRG-100 with LOS > 25 and NO major complications or secondary procedures indicates overpayment.",
     "defined_by": "SME001", "triggers_overpayment_category": "Long_Stay_No_Justification"}
]

# Simulate a CatBoost model "insight" - a claim flagged as overpayment
catboost_flags = ["C1002", "C1004"] # These are the claims CatBoost identified as overpayment


print("Ingesting data into the conceptual Knowledge Graph...")

# Ingest DRG nodes and their typical values/associations
for drg_code, props in drg_definitions.items():
    drg_node_id = f"DRG_{drg_code}"
    add_node(drg_node_id, "DRG", properties={
        "code": drg_code,
        "typical_los_min": props["typical_los_min"],
        "typical_los_max": props["typical_los_max"]
    })

    # Link typical procedures to DRGs
    for proc_code in props["typical_procedures"]:
        proc_node_id = f"PROC_{proc_code}"
        add_node(proc_node_id, "Procedure", {"code": proc_code})
        add_edge(drg_node_id, "TYPICALLY_INCLUDES_PROCEDURE", proc_node_id)

    # Link common diagnoses to DRGs
    for diag_code in props["common_diagnoses"]:
        diag_node_id = f"DIAG_{diag_code}"
        add_node(diag_node_id, "Diagnosis", {"code": diag_code})
        add_edge(drg_node_id, "COMMONLY_HAS_DIAGNOSIS", diag_node_id)

    # Link major complications to DRGs (these are also diagnoses, but with a special role)
    for comp_code in props["major_complications"]:
        comp_node_id = f"DIAG_{comp_code}"
        # Add "Complication" label to diagnosis nodes that are also complications
        add_node(comp_node_id, ["Diagnosis", "Complication"], {"code": comp_code})
        add_edge(drg_node_id, "HAS_MAJOR_COMPLICATION_POSSIBLY", comp_node_id)

# Ingest data from your `your_dataset` DataFrame (Claims and related entities)
for index, row in your_dataset.iterrows():
    claim_id = f"CLAIM_{row['claim_id']}"
    provider_id = f"PROVIDER_{row['provider_id']}"
    drg_id = f"DRG_{row['drg_code']}"
    discharge_status_id = f"DISP_{row['discharge_status'].replace(' ', '_')}"
    principal_diag_id = f"DIAG_{row['principal_diagnosis']}"

    # Add nodes for claim and its direct properties
    add_node(claim_id, "Claim", {
        "id": row['claim_id'],
        "los": row['los'],
        # Integrate CatBoost flag directly as a property of the claim
        "catboost_flagged_overpayment": row['claim_id'] in catboost_flags
    })
    add_node(provider_id, "Provider", {"id": row['provider_id']})
    add_node(discharge_status_id, "DischargeStatus", {"type": row['discharge_status']})
    add_node(principal_diag_id, "Diagnosis", {"code": row['principal_diagnosis']})

    # Add relationships for principal claim data
    add_edge(claim_id, "BILLED_BY", provider_id)
    add_edge(claim_id, "ASSIGNED_TO_DRG", drg_id)
    add_edge(claim_id, "HAS_DISCHARGE_STATUS", discharge_status_id)
    add_edge(claim_id, "HAS_PRINCIPAL_DIAGNOSIS", principal_diag_id)

    # Handle comorbidities (now mapped as secondary diagnoses)
    if pd.notna(row['comorbidities']) and row['comorbidities']:
        for diag_code in str(row['comorbidities']).split(','):
            diag_code = diag_code.strip()
            if diag_code: # Ensure not empty string from split
                comorb_diag_id = f"DIAG_{diag_code}"
                add_node(comorb_diag_id, "Diagnosis", {"code": diag_code})
                add_edge(claim_id, "HAS_SECONDARY_DIAGNOSIS", comorb_diag_id)

    # Handle procedures
    if pd.notna(row['procedures']) and row['procedures']:
        for proc_code in str(row['procedures']).split(','):
            proc_code = proc_code.strip()
            if proc_code:
                proc_node_id = f"PROC_{proc_code}"
                add_node(proc_node_id, "Procedure", {"code": proc_code})
                add_edge(claim_id, "HAS_PROCEDURE", proc_node_id)


# Ingest SME and Policy nodes and rules
# Ensure SME and Policy nodes exist before linking rules
sme_node_id = add_node("SME_SME001", "SME", {"id": "SME001", "name": "Dr. Sarah Lee"})
policy_node_id = add_node("POLICY_POLICY001", "Policy", {"id": "POLICY001", "name": "CMS Chapter 3 - Inpatient Payment"})

for rule_dict in sme_rules:
    rule_id = f"RULE_{rule_dict['rule_id']}"
    drg_id = f"DRG_{rule_dict['drg_code']}"

    add_node(rule_id, "Rule", {
        "id": rule_dict["rule_id"],
        "description": rule_dict["description"],
        "los_threshold": rule_dict["los_threshold"],
        "triggers_overpayment_category": rule_dict["triggers_overpayment_category"]
    })
    add_edge(rule_id, "APPLIES_TO_DRG", drg_id)
    add_edge(rule_id, "DEFINED_BY_SME", sme_node_id)
    add_edge(rule_id, "IS_BASED_ON_POLICY", policy_node_id)

print("Data ingestion complete.")
print(f"Total nodes: {len(kg_nodes)}")
print(f"Total edges: {len(kg_edges)}")

# --- 3. Querying for Concepts and Overpayment Identification ---

# Helper function to find connected nodes via a specific relationship type from a source node
def get_connected_nodes(source_id, relationship_type, target_label=None):
    results = []
    for s, r_type, t in kg_edges:
        if s == source_id and r_type == relationship_type:
            # Optionally filter by target node label
            if target_label:
                if target_label in kg_nodes[t]['labels']:
                    results.append(kg_nodes[t])
            else:
                results.append(kg_nodes[t])
    return results

# Helper function to find source nodes connected to a target node via a specific relationship type
def get_sources_connected_to(target_id, relationship_type, source_label=None):
    results = []
    for s, r_type, t in kg_edges:
        if t == target_id and r_type == relationship_type:
            # Optionally filter by source node label
            if source_label:
                if source_label in kg_nodes[s]['labels']:
                    results.append(kg_nodes[s])
            else:
                results.append(kg_nodes[s])
    return results

print("\n--- Simulating Queries ---")

# --- Scenario 1: Query a DRG and get its related diagnosis codes, procedures, and associated overpayment rules ---
def query_drg_info(drg_code_to_query):
    print(f"\n--- Querying DRG: {drg_code_to_query} ---")
    drg_node_id = f"DRG_{drg_code_to_query}"

    if drg_node_id not in kg_nodes:
        print(f"DRG {drg_code_to_query} not found in KG.")
        return

    drg_props = kg_nodes[drg_node_id]['properties']
    print(f"DRG Code: {drg_props['code']}")
    print(f"Typical LOS Range: {drg_props['typical_los_min']} - {drg_props['typical_los_max']}")

    # Get related diagnoses (those commonly associated with this DRG)
    common_diagnoses = get_connected_nodes(drg_node_id, "COMMONLY_HAS_DIAGNOSIS", "Diagnosis")
    print(f"Common Diagnoses for this DRG: {[d['properties']['code'] for d in common_diagnoses]}")

    # Get major complications known for this DRG (might justify extended LOS)
    major_complications_for_drg = get_connected_nodes(drg_node_id, "HAS_MAJOR_COMPLICATION_POSSIBLY", "Complication")
    print(f"Major Complications known for this DRG: {[c['properties']['code'] for c in major_complications_for_drg]}")

    # Get related procedures (those typically included for this DRG)
    typical_procedures = get_connected_nodes(drg_node_id, "TYPICALLY_INCLUDES_PROCEDURE", "Procedure")
    print(f"Typical Procedures for this DRG: {[p['properties']['code'] for p in typical_procedures]}")

    # Get overpayment rules specifically applicable to this DRG
    applicable_rules = get_sources_connected_to(drg_node_id, "APPLIES_TO_DRG", "Rule")
    print("\nApplicable Overpayment Rules:")
    if applicable_rules:
        for rule in applicable_rules:
            rule_props = rule['properties']
            print(f"- Rule ID: {rule_props['id']}")
            print(f"  Description: {rule_props['description']}")
            print(f"  LOS Threshold: {rule_props.get('los_threshold', 'N/A')}")
            print(f"  Triggers Category: {rule_props.get('triggers_overpayment_category', 'N/A')}")
    else:
        print("  No explicit overpayment rules found for this DRG.")

# Test Scenario 1
query_drg_info("DRG-100")
query_drg_info("DRG-200")
query_drg_info("DRG-300") # Test non-existent DRG

# --- Scenario 2: Query a specific Claim and identify potential overpayment reasons ---
def analyze_claim_for_overpayment(claim_id_to_analyze):
    print(f"\n--- Analyzing Claim: {claim_id_to_analyze} for Overpayment ---")
    claim_node_id = f"CLAIM_{claim_id_to_analyze}"

    if claim_node_id not in kg_nodes:
        print(f"Claim {claim_id_to_analyze} not found in KG.")
        return

    claim_props = kg_nodes[claim_node_id]['properties']
    claim_los = claim_props['los']
    print(f"Claim ID: {claim_props['id']}, LOS: {claim_los}")

    # Get associated DRG details
    assigned_drg_nodes = get_connected_nodes(claim_node_id, "ASSIGNED_TO_DRG", "DRG")
    if not assigned_drg_nodes:
        print("No DRG assigned to this claim.")
        return

    drg_node_data = assigned_drg_nodes[0]
    drg_code = drg_node_data['properties']['code']
    drg_typical_los_min = drg_node_data['properties']['typical_los_min']
    drg_typical_los_max = drg_node_data['properties']['typical_los_max']
    print(f"Assigned DRG: {drg_code} (Typical LOS: {drg_typical_los_min}-{drg_typical_los_max})")

    # Get Principal and Secondary Diagnoses (Comorbidities from your dataset)
    principal_diag = get_connected_nodes(claim_node_id, "HAS_PRINCIPAL_DIAGNOSIS", "Diagnosis")
    secondary_diagnoses = get_connected_nodes(claim_node_id, "HAS_SECONDARY_DIAGNOSIS", "Diagnosis")
    all_diagnoses_on_claim = [d['properties']['code'] for d in (principal_diag + secondary_diagnoses)]
    print(f"Diagnoses on Claim: {all_diagnoses_on_claim}")

    # Get Procedures on Claim
    procedures_on_claim = get_connected_nodes(claim_node_id, "HAS_PROCEDURE", "Procedure")
    print(f"Procedures on Claim: {[p['properties']['code'] for p in procedures_on_claim]}")
    
    # Get Provider ID
    billed_by_provider = get_connected_nodes(claim_node_id, "BILLED_BY", "Provider")
    if billed_by_provider:
        print(f"Billed by Provider: {billed_by_provider[0]['properties']['id']}")

    print("\n--- Overpayment Analysis ---")
    overpayment_reasons = []

    # 1. Check against SME-defined rules (automated rule application)
    applicable_rules_for_drg = get_sources_connected_to(f"DRG_{drg_code}", "APPLIES_TO_DRG", "Rule")
    for rule in applicable_rules_for_drg:
        rule_props = rule['properties']
        rule_id = rule_props['id']
        rule_description = rule_props['description']
        rule_los_threshold = rule_props.get('los_threshold')
        overpayment_category = rule_props.get('triggers_overpayment_category')

        if rule_id == "RULE_DRG100_LOS_Short_NoComorbid":
            if claim_los < rule_los_threshold:
                # Check for major comorbidities (secondary diagnoses linked as complications for this DRG)
                has_major_comorbid_on_claim = False
                major_complications_for_drg = get_connected_nodes(f"DRG_{drg_code}", "HAS_MAJOR_COMPLICATION_POSSIBLY", "Complication")
                major_comp_codes = [c['properties']['code'] for c in major_complications_for_drg]
                
                for diag_node in secondary_diagnoses:
                    if diag_node['properties']['code'] in major_comp_codes:
                        has_major_comorbid_on_claim = True
                        break
                
                if not has_major_comorbid_on_claim:
                    overpayment_reasons.append(f"VIOLATES_SME_RULE: '{rule_id}' - {rule_description} (Category: {overpayment_category})")
        
        elif rule_id == "RULE_DRG100_LOS_Long_NoJustify":
            if claim_los > rule_los_threshold:
                # Check for justification (e.g., major complications present on the claim)
                has_justification_on_claim = False
                major_complications_for_drg = get_connected_nodes(f"DRG_{drg_code}", "HAS_MAJOR_COMPLICATION_POSSIBLY", "Complication")
                major_comp_codes = [c['properties']['code'] for c in major_complications_for_drg]

                for diag_node in secondary_diagnoses: # Check secondary diagnoses for justification
                    if diag_node['properties']['code'] in major_comp_codes:
                        has_justification_on_claim = True
                        break
                
                if not has_justification_on_claim:
                    overpayment_reasons.append(f"VIOLATES_SME_RULE: '{rule_id}' - {rule_description} (Category: {overpayment_category})")


    # 2. Check against typical DRG patterns (data-driven anomaly detection)
    # This identifies patterns that might not have a hard SME rule yet.
    if claim_los < drg_typical_los_min:
        # If LOS is too low, check if common diagnoses for the DRG are missing on the claim
        lacks_common_diag_on_claim = True
        common_diagnoses_for_drg = get_connected_nodes(f"DRG_{drg_code}", "COMMONLY_HAS_DIAGNOSIS", "Diagnosis")
        common_diag_codes = [d['properties']['code'] for d in common_diagnoses_for_drg]
        
        for diag_code in all_diagnoses_on_claim:
            if diag_code in common_diag_codes:
                lacks_common_diag_on_claim = False
                break

        # If very short LOS, no common diag, and no secondary diag (comorbidity) to explain complexity
        if lacks_common_diag_on_claim and not secondary_diagnoses:
            overpayment_reasons.append("DATA_ANOMALY: Claim LOS is significantly below DRG typical minimum, and lacks common diagnoses or any secondary diagnoses to explain complexity.")

    if claim_los > drg_typical_los_max:
        # If LOS is too high, check if it lacks documented major complications to justify extended stay
        lacks_major_complication_justification_on_claim = True
        major_complications_for_drg = get_connected_nodes(f"DRG_{drg_code}", "HAS_MAJOR_COMPLICATION_POSSIBLY", "Complication")
        major_comp_codes = [c['properties']['code'] for c in major_complications_for_drg]

        for diag_node in secondary_diagnoses:
            if diag_node['properties']['code'] in major_comp_codes:
                lacks_major_complication_justification_on_claim = False
                break
        
        if lacks_major_complication_justification_on_claim:
            overpayment_reasons.append("DATA_ANOMALY: Claim LOS is significantly above DRG typical maximum, but lacks documented major complications to justify extended stay.")


    # 3. Integrate CatBoost Model Flag (if available)
    if claim_props.get('catboost_flagged_overpayment'):
        overpayment_reasons.append("ML_MODEL_FLAG: CatBoost model identified this claim as potential overpayment.")


    if overpayment_reasons:
        print("\n**Potential Overpayment Identified! Reasons:**")
        for reason in overpayment_reasons:
            print(f"- {reason}")
    else:
        print("No immediate overpayment flags or anomalies detected based on current rules and knowledge.")

# Test Scenario 2
analyze_claim_for_overpayment("C1001") # Expected: No overpayment
analyze_claim_for_overpayment("C1002") # Expected: Flagged by SME rule and data anomaly (short LOS)
analyze_claim_for_overpayment("C1004") # Expected: Flagged by CatBoost, and potentially by data anomaly (long LOS) if comorbidities don't justify
analyze_claim_for_overpayment("C1003") # Expected: No overpayment (within typical LOS range)
analyze_claim_for_overpayment("C9999") # Expected: Non-existent claim

print("\nScript finished.")
