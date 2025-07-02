import pandas as pd

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
        return f"DRG_Combined_{'_'.join(sorted(diag_list))}_P{'_'.join(sorted(proc_list))}"

def generate_drg_on_proc_removal(row):
    """
    Generates a dictionary of DRGs after removing each procedure code.
    Assumes 'diag_codes' and 'proc_codes' columns exist in the row.
    """
    original_diag_codes = row['diag_codes']
    original_proc_codes = row['proc_codes']

    # Initialize the dictionary to store results
    drg_results = {}

    # Handle the case where there are no procedure codes to remove
    if not original_proc_codes:
        # You might want to define a specific DRG for this scenario,
        # or simply return an empty dict, or the DRG with no proc codes.
        drg_results['No_Proc_Codes_Originally'] = calculate_drg(original_diag_codes, [])
        return drg_results

    # Iterate through each proc code to see the effect of its removal
    for i, proc_code_to_remove in enumerate(original_proc_codes):
        # Create a new list of proc codes, excluding the current one
        modified_proc_codes = original_proc_codes[:i] + original_proc_codes[i+1:]

        # Calculate the DRG with the modified proc codes
        new_drg = calculate_drg(original_diag_codes, modified_proc_codes)

        # Add to the dictionary: key = removed proc code, value = new DRG
        drg_results[proc_code_to_remove] = new_drg

    return drg_results

# --- Example Usage ---

# Create a sample DataFrame
data = {
    'patient_id': [1, 2, 3],
    'diag_codes': [['D1', 'D2'], ['D3'], ['D4', 'D5', 'D6']],
    'proc_codes': [['A', 'B', 'C'], ['X', 'Y'], []] # Example with empty proc list
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
print("\n" + "="*50 + "\n")

# Apply the function to create the new column
df['drg_on_proc_removal'] = df.apply(generate_drg_on_proc_removal, axis=1)

print("DataFrame with new 'drg_on_proc_removal' column:")
print(df)

# To inspect the content of the new column for a specific row:
# print(f"\nDRG possibilities for patient 1:\n{df.loc[0, 'drg_on_proc_removal']}")
