import pandas as pd
import numpy as np # For np.nan for nulls

# Assume you have your DRG calculation function
# IMPORTANT: REPLACE THIS WITH YOUR ACTUAL, ORDER-SENSITIVE DRG GROUPING LOGIC
# For demonstration, this version is made sensitive to the primary diagnosis (diag_list[0])
def calculate_drg(diag_list, proc_list):
    """
    Simulates a DRG calculation function.
    In a real scenario, this would be your complex DRG grouping logic.
    This demo version is sensitive to the primary diagnosis (diag_list[0]).
    """
    diag_list = diag_list if diag_list is not None else []
    proc_list = proc_list if proc_list is not None else []

    pdx = diag_list[0] if diag_list else None # Primary diagnosis

    # Handle empty lists gracefully
    if not diag_list and not proc_list:
        return "DRG_NoCodes"
    if not diag_list:
        return f"DRG_ProcOnly_{'_'.join(sorted(proc_list))}"
    if not proc_list:
        if pdx == 'D1': return "DRG_DIAG_D1_NoProc"
        if pdx == 'D2': return "DRG_DIAG_D2_NoProc"
        if pdx == 'D3': return "DRG_DIAG_D3_NoProc"
        return f"DRG_DiagOnly_{pdx}"

    # Rules sensitive to PDX and procedures
    if pdx == 'D1' and 'A' in proc_list and 'B' in proc_list:
        return "DRG_MAJOR_HEART_SURG" # High weight, PDX D1, Procs A & B
    if pdx == 'D2' and 'A' in proc_list and 'B' in proc_list:
        return "DRG_MINOR_HEART_SURG" # Lower weight if D2 is PDX
    if pdx == 'D1' and 'C' in proc_list:
        return "DRG_GASTRO_PROC"
    if pdx == 'D3' and 'X' in proc_list and 'Y' in proc_list:
        return "DRG_NEURO_COMPLEX"
    if pdx == 'D4' and 'Z' in proc_list:
        return "DRG_RESP_LITE"
    if pdx == 'D5' and 'A' in proc_list: # Specific case for D5 PDX
        return "DRG_RENAL_ISSUE"

    # Fallback for other combinations
    # Using sorted list for robustness in fallback DRG names
    return f"DRG_OTHER_PDX_{pdx or 'NoPDX'}_PROCS_{'_'.join(sorted(proc_list))}"


def generate_drg_on_diag_modifications(row):
    """
    Generates a dictionary of DRGs after applying various diagnosis code modifications:
    1. Removal of each single diagnosis code.
    2. Switching the primary diagnosis (PDX) with another diagnosis in the list.
    Assumes 'diag_codes' and 'proc_codes' columns exist in the row.
    """
    original_diag_codes = row['diag_codes'] if isinstance(row['diag_codes'], list) else []
    original_proc_codes = row['proc_codes'] if isinstance(row['proc_codes'], list) else []

    drg_results = {}

    # --- Scenario 1: Removal of a single diagnosis code ---
    if original_diag_codes:
        for i, diag_code_to_remove in enumerate(original_diag_codes):
            modified_diag_codes_for_removal = original_diag_codes[:i] + original_diag_codes[i+1:]
            
            # Ensure the remaining list isn't empty for calculate_drg, pass empty list if so
            new_drg = calculate_drg(modified_diag_codes_for_removal, original_proc_codes)
            drg_results[f"removed_diag_{diag_code_to_remove}"] = new_drg

    # --- Scenario 2: Switching Primary Diagnosis (PDX) ---
    # This applies if there's more than one diagnosis to consider for PDX
    if len(original_diag_codes) > 1:
        current_pdx = original_diag_codes[0]
        
        for diag_code_candidate_pdx in original_diag_codes:
            if diag_code_candidate_pdx != current_pdx:
                # Create a new list where candidate is first, others follow
                # This ensures the new PDX is at index 0, and others retain relative order
                modified_diag_codes_for_pdx = [diag_code_candidate_pdx] + \
                                             [d for d in original_diag_codes if d != diag_code_candidate_pdx]
                
                new_drg = calculate_drg(modified_diag_codes_for_pdx, original_proc_codes)
                drg_results[f"pdx_switched_to_{diag_code_candidate_pdx}"] = new_drg
    
    return drg_results if drg_results else np.nan # Return np.nan if no modifications could be applied or no results


# --- NEW: Incorporating your drg_df ---

# Create a sample drg_df (REPLACE WITH YOUR ACTUAL drg_df)
# I've added DRGs relevant to the new calculate_drg logic and made weights to demonstrate changes
drg_data = {
    'DRG_Code': [
        "DRG_MAJOR_HEART_SURG", "DRG_MINOR_HEART_SURG", "DRG_GASTRO_PROC",
        "DRG_NEURO_COMPLEX", "DRG_RESP_LITE", "DRG_RENAL_ISSUE",
        "DRG_NoCodes", "DRG_ProcOnly_A_B", "DRG_ProcOnly_X_Y", "DRG_ProcOnly_Z",
        "DRG_DIAG_D1_NoProc", "DRG_DIAG_D2_NoProc", "DRG_DIAG_D3_NoProc",
        "DRG_DiagOnly_D4", "DRG_DiagOnly_D5",
        "DRG_OTHER_PDX_D1_PROCS_AB", "DRG_OTHER_PDX_D2_PROCS_AB",
        "DRG_OTHER_PDX_D3_PROCS_XY"
    ],
    'Weight': [
        2.800,  # DRG_MAJOR_HEART_SURG
        1.200,  # DRG_MINOR_HEART_SURG (significantly lower than MAJOR)
        1.500,
        3.000,
        0.900,
        1.100,  # DRG_RENAL_ISSUE
        0.100, 0.500, 2.500, 0.700, # Base cases
        0.200, 0.150, 0.250, # No proc, diag only
        0.300, 0.400,
        2.700, 1.150, 2.900 # Other DRGs for fallbacks
    ]
}
drg_df = pd.DataFrame(drg_data)

# Create a mapping for quick lookup: DRG_Code -> Weight
drg_weight_map = drg_df.set_index('DRG_Code')['Weight'].to_dict()

def get_drg_weight(drg_code, weight_map):
    """Safely gets the weight for a DRG code from the map. Returns inf if not found."""
    return weight_map.get(str(drg_code), float('inf')) # Ensure drg_code is string for lookup

def find_lower_weight_drgs(row, weight_map):
    """
    Checks the DRGs in 'drg_on_diag_modifications' and returns a dictionary
    of those with weights lower than the billed DRG, or None if none.
    """
    drg_possibilities = row['drg_on_diag_modifications']
    
    # If drg_possibilities is NaN (no modifications possible/generated), return NaN directly
    if pd.isna(drg_possibilities):
        return np.nan

    original_billed_drg = row['billed_drg']
    current_billed_drg_weight = get_drg_weight(original_billed_drg, weight_map)

    lower_weight_drgs = {}

    for modification_key, new_drg_code in drg_possibilities.items():
        new_drg_weight = get_drg_weight(new_drg_code, weight_map)

        # Compare the new DRG's weight to the original billed DRG's weight for this row
        if new_drg_weight < current_billed_drg_weight:
            lower_weight_drgs[modification_key] = new_drg_code

    if lower_weight_drgs:
        return lower_weight_drgs
    else:
        return np.nan # Use pandas' NaN for nulls


# --- Example Usage ---

# Create a sample DataFrame (patient data)
patient_data = {
    'patient_id': [1, 2, 3, 4, 5, 6],
    'diag_codes': [
        ['D1', 'D2', 'D3'],     # Case 1: Multiple diags, PDX D1 (high DRG if A,B procs)
        ['D2', 'D1'],           # Case 2: D2 is PDX, but D1 could be PDX for higher DRG
        ['D3'],                 # Case 3: Single diag, removal means no diag
        ['D4', 'D5'],           # Case 4: Multiple diags, check PDX switches and removals
        ['D1', 'D5'],           # Case 5: D1 PDX, D5 secondary
        ['D5', 'D1']            # Case 6: D5 PDX, D1 secondary (should be lower DRG if procs A)
    ],
    'proc_codes': [
        ['A', 'B'],             # Case 1: D1 PDX + A,B procs -> MAJOR_HEART_SURG (2.8)
        ['A', 'B'],             # Case 2: D2 PDX + A,B procs -> MINOR_HEART_SURG (1.2)
        [],                     # Case 3: No procs
        ['Z'],                  # Case 4: D4 PDX + Z proc -> RESP_LITE (0.9)
        ['A'],                  # Case 5: D1 PDX + A proc -> Fallback DRG (e.g., OTHER_PDXD1_PROCS_A)
        ['A']                   # Case 6: D5 PDX + A proc -> RENAL_ISSUE (1.1)
    ],
    'billed_drg': [
        'DRG_MAJOR_HEART_SURG', # For patient 1, billed DRG is high
        'DRG_MINOR_HEART_SURG', # For patient 2, billed DRG is lower (D2 PDX)
        'DRG_DIAG_D3_NoProc',   # Patient 3
        'DRG_RESP_LITE',        # Patient 4
        'DRG_OTHER_PDX_D1_PROCS_A', # Patient 5
        'DRG_RENAL_ISSUE'       # Patient 6
    ]
}
df = pd.DataFrame(patient_data)

print("Original Patient DataFrame:")
print(df)
print("\n" + "="*50 + "\n")

print("DRG Weights DataFrame (for lookup):")
print(drg_df.to_string()) # Use to_string() to see full DRG names if they are long
print("\n" + "="*50 + "\n")

# Step 1: Generate the column with DRGs on diag modifications
df['drg_on_diag_modifications'] = df.apply(generate_drg_on_diag_modifications, axis=1)

print("DataFrame after generating 'drg_on_diag_modifications' (raw results):")
print(df.to_string()) # Use to_string() to see full content of columns
print("\n" + "="*50 + "\n")

# Step 2: Create the new column based on weight comparison using the drg_weight_map
df['lower_weight_drgs_after_modifications'] = df.apply(
    lambda row: find_lower_weight_drgs(row, drg_weight_map), axis=1
)

print("Final DataFrame with 'lower_weight_drgs_after_modifications' column:")
print(df[['patient_id', 'diag_codes', 'proc_codes', 'billed_drg', 'lower_weight_drgs_after_modifications']].to_string())
print("\n" + "="*50 + "\n")

print("Explanation of selected example outcomes:")
print("Patient 1: Billed DRG is DRG_MAJOR_HEART_SURG (2.8).")
print("  If PDX switches to D2, it becomes DRG_MINOR_HEART_SURG (1.2), which is lower.")
print("  This scenario will be captured in 'lower_weight_drgs_after_modifications'.")
print("Patient 2: Billed DRG is DRG_MINOR_HEART_SURG (1.2).")
print("  Original PDX is D2. If PDX switches to D1, it becomes DRG_MAJOR_HEART_SURG (2.8), which is higher.")
print("  No lower weight DRGs will be found for this patient.")
print("Patient 6: Billed DRG is DRG_RENAL_ISSUE (1.1). (PDX D5, Proc A)")
print("  If PDX switches to D1, it becomes DRG_OTHER_PDX_D1_PROCS_A (fallback, assumed higher or equal).")
print("  If 'D5' is removed, and 'D1' becomes the only diag, it would become DRG_DIAG_D1_NoProc (0.2), which is lower.")
print("  This scenario will also be captured.")

