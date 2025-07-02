import pandas as pd

# --- Assume you have your DataFrame and DRG calculation function ---

# Example DataFrame (replace with your actual data)
data = {'diag_codes': [['D1', 'D2', 'D3']],
        'proc_codes': [['A', 'B', 'C']]}
df = pd.DataFrame(data)

# Placeholder for your actual DRG calculation function
# In reality, this would be a complex function based on DRG grouping logic
def calculate_drg(diag_list, proc_list):
    """
    Simulates a DRG calculation function.
    Replace this with your actual DRG logic.
    """
    if not proc_list: # If no proc codes, just a basic DRG based on diag
        return f"DRG_Base_{'_'.join(diag_list)}"
    else:
        return f"DRG_{'_'.join(diag_list)}_Proc_{'_'.join(proc_list)}"

# --- Main Logic ---

# Let's consider the first row of your DataFrame for this example
# You might iterate through rows if you want to do this for multiple cases
row_index = 0
original_diag_codes = df.loc[row_index, 'diag_codes']
original_proc_codes = df.loc[row_index, 'proc_codes']

print(f"Original Diagnosis Codes: {original_diag_codes}")
print(f"Original Procedure Codes: {original_proc_codes}\n")

# Calculate the DRG with all original codes
original_drg = calculate_drg(original_diag_codes, original_proc_codes)
print(f"DRG with all original codes: {original_drg}\n")

print("Possible DRGs after removing one procedure code:")
possible_drgs_after_removal = []

# Iterate through each proc code to see the effect of its removal
for i, proc_code_to_remove in enumerate(original_proc_codes):
    # Create a new list of proc codes, excluding the current one
    modified_proc_codes = original_proc_codes[:i] + original_proc_codes[i+1:]

    # Calculate the DRG with the modified proc codes
    new_drg = calculate_drg(original_diag_codes, modified_proc_codes)

    possible_drgs_after_removal.append({
        'removed_proc_code': proc_code_to_remove,
        'remaining_proc_codes': modified_proc_codes,
        'new_drg': new_drg
    })
    print(f"  - Removed '{proc_code_to_remove}': Remaining Procs: {modified_proc_codes}, New DRG: {new_drg}")

print("\nSummary of possible DRGs after removal:")
for result in possible_drgs_after_removal:
    print(f"  Removed: {result['removed_proc_code']}, New DRG: {result['new_drg']}")

# If you want to store these results back into the DataFrame, you could do so
# For example, add a new column with a list of these possibilities
# df.loc[row_index, 'drg_possibilities_on_proc_removal'] = possible_drgs_after_removal
