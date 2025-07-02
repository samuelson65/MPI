import pandas as pd
import numpy as np # For np.nan for nulls

# Assume you have your DRG calculation function
# REPLACE THIS WITH YOUR ACTUAL DRG LOGIC
def calculate_drg(diag_list, proc_list):
    """
    Simulates a DRG calculation function.
    In a real scenario, this would be your complex DRG grouping logic.
    """
    if not proc_list and not diag_list:
        return "DRG_NoCodes"
    elif not proc_list:
        return f"DRG_DiagOnly_{'_'.join(sorted(diag_list))}"
    elif not diag_list:
        return f"DRG_ProcOnly_{'_'.join(sorted(proc_list))}"
    else:
        # A more realistic DRG output for the example
        # Let's make it produce different DRGs to illustrate weight comparison
        # These DRGs should match codes in your drg_df for weight lookup
        if 'A' in proc_list and 'D1' in diag_list:
            return "DRG_481"
        elif 'B' in proc_list:
            return "DRG_291"
        elif 'C' in proc_list:
            return "DRG_101"
        elif not proc_list and 'D3' in diag_list:
            return "DRG_999" # Example for a case with no procs
        elif 'X' in proc_list and 'Y' in proc_list:
            return "DRG_500"
        elif 'Z' in proc_list:
            return "DRG_600"
        else:
            # Fallback for other combinations, ensures a DRG code is returned
            return f"DRG_Other_{len(proc_list)}_{''.join(sorted(proc_list or ['NoP']))}"


def generate_drg_on_proc_removal(row):
    """
    Generates a dictionary of DRGs after removing each procedure code.
    Assumes 'diag_codes' and 'proc_codes' columns exist in the row.
    """
    original_diag_codes = row['diag_codes']
    original_proc_codes = row['proc_codes']

    drg_results = {}

    if not original_proc_codes:
        drg_results['No_Proc_Codes_Originally'] = calculate_drg(original_diag_codes, [])
        return drg_results

    for i, proc_code_to_remove in enumerate(original_proc_codes):
        modified_proc_codes = original_proc_codes[:i] + original_proc_codes[i+1:]
        new_drg = calculate_drg(original_diag_codes, modified_proc_codes)
        drg_results[proc_code_to_remove] = new_drg

    return drg_results

# --- NEW: Incorporating your drg_df ---

# Create a sample drg_df (REPLACE WITH YOUR ACTUAL drg_df)
drg_data = {
    'DRG_Code': ["DRG_481", "DRG_291", "DRG_101", "DRG_999", "DRG_500", "DRG_600",
                 "DRG_NoCodes", "DRG_DiagOnly_D4_D5_D6", "DRG_Other_2_AB", "DRG_Other_1_A",
                 "DRG_Other_1_C", "DRG_Other_0_NoP"],
    'Weight': [2.4819, 1.5000, 0.8000, 0.5000, 3.5000, 1.2000, 0.1000, 0.7000, 2.0000, 1.0000, 0.9000, 0.3000]
}
drg_df = pd.DataFrame(drg_data)

# Create a mapping for quick lookup: DRG_Code -> Weight
drg_weight_map = drg_df.set_index('DRG_Code')['Weight'].to_dict()

def get_drg_weight(drg_code, weight_map):
    """Safely gets the weight for a DRG code from the map. Returns inf if not found."""
    return weight_map.get(str(drg_code), float('inf')) # Ensure drg_code is string for lookup

def find_lower_weight_drgs(row, weight_map):
    """
    Checks the DRGs in 'drg_on_proc_removal' and returns a dictionary
    of those with weights lower than the billed DRG, or None if none.
    """
    drg_possibilities = row['drg_on_proc_removal']
    original_billed_drg = row['billed_drg']

    # Get the weight of the actual billed DRG for this row using the provided map
    current_billed_drg_weight = get_drg_weight(original_billed_drg, weight_map)

    lower_weight_drgs = {}

    for removed_proc_code, new_drg_code in drg_possibilities.items():
        new_drg_weight = get_drg_weight(new_drg_code, weight_map)

        # Compare the new DRG's weight to the original billed DRG's weight for this row
        if new_drg_weight < current_billed_drg_weight:
            lower_weight_drgs[removed_proc_code] = new_drg_code

    if lower_weight_drgs:
        return lower_weight_drgs
    else:
        return np.nan # Use pandas' NaN for nulls


# --- Example Usage ---

# Create a sample DataFrame (patient data)
patient_data = {
    'patient_id': [1, 2, 3, 4, 5],
    'diag_codes': [['D1', 'D2'], ['D3'], ['D4', 'D5', 'D6'], ['D7'], ['D8']],
    'proc_codes': [['A', 'B', 'C'], ['X', 'Y'], [], ['Z'], ['A']], # Added a case where removing 'A' leads to lower DRG
    'billed_drg': ['DRG_481', 'DRG_500', 'DRG_DiagOnly_D4_D5_D6', 'DRG_600', 'DRG_481']
}
df = pd.DataFrame(patient_data)

print("Original Patient DataFrame:")
print(df)
print("\n" + "="*50 + "\n")

print("DRG Weights DataFrame:")
print(drg_df)
print("\n" + "="*50 + "\n")

# Step 1: Generate the column with DRGs on proc removal
df['drg_on_proc_removal'] = df.apply(generate_drg_on_proc_removal, axis=1)

print("DataFrame after generating 'drg_on_proc_removal':")
print(df)
print("\n" + "="*50 + "\n")

# Step 2: Create the new column based on weight comparison using the drg_weight_map
# Pass the drg_weight_map to the apply function using a lambda or functools.partial
df['lower_weight_drgs_after_removal'] = df.apply(
    lambda row: find_lower_weight_drgs(row, drg_weight_map), axis=1
)

print("Final DataFrame with 'lower_weight_drgs_after_removal' column:")
print(df.to_string()) # Use to_string() to see full content of columns for better display
