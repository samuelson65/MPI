import pandas as pd
import numpy as np

# ==============================================================================
#  YOUR DATA LOADING SECTION
#  IMPORTANT: Place your code to load the DataFrame 'df' here.
#  The rest of the script assumes 'df' exists with columns:
#  - 'Diag soi' (dictionary of diag codes and soi values)
#  - 'aprdrg' (4-digit code)
#  - 'sta' ('I' for overpayment, 'Z' for no findings)
# ==============================================================================
# Example: df = pd.read_csv('your_data.csv')
# Example: df = pd.read_excel('your_data.xlsx')

# ==============================================================================
#  PREPROCESSING: FEATURE EXTRACTION (Same as previous script)
# ==============================================================================

def extract_diag_features(diag_dict):
    """Safely extracts number of diagnoses and max SOI from a dictionary."""
    try:
        if not isinstance(diag_dict, dict) or not diag_dict:
            return 0, 0
        soi_values = list(diag_dict.values())
        return len(soi_values), max(soi_values)
    except Exception:
        return 0, 0

df[['num_diags', 'max_soi']] = df['Diag soi'].apply(
    lambda x: pd.Series(extract_diag_features(x))
)

df['aprdrg'] = df['aprdrg'].astype(str).str.zfill(4)
df['aprdrg_drg'] = df['aprdrg'].str[:3]
df['aprdrg_soi'] = df['aprdrg'].str[3].astype(int)

df['soi_discrepancy'] = (df['aprdrg_soi'] != df['max_soi'])


# ==============================================================================
#  INSIGHTS GENERATION
#  This section calculates and prints meaningful insights from the data.
# ==============================================================================

print("="*80)
print("             INSIGHTS FROM OVERPAYMENT DATA")
print("="*80)

# --- Overall Summary ---
total_claims = len(df)
overpayment_claims = (df['sta'] == 'I').sum()
overpayment_rate = overpayment_claims / total_claims * 100

print("\n### 1. Overall Dataset Summary")
print(f"Total claims analyzed: {total_claims:,}")
print(f"Claims with overpayment ('I'): {overpayment_claims:,}")
print(f"Claims with no findings ('Z'): {(df['sta'] == 'Z').sum():,}")
print(f"Overall overpayment rate: {overpayment_rate:.2f}%")


# --- Insights from Clinical Data ('Diag soi') ---
avg_num_diags_I = df[df['sta'] == 'I']['num_diags'].mean()
avg_num_diags_Z = df[df['sta'] == 'Z']['num_diags'].mean()
avg_max_soi_I = df[df['sta'] == 'I']['max_soi'].mean()
avg_max_soi_Z = df[df['sta'] == 'Z']['max_soi'].mean()

print("\n### 2. Clinical and Severity Insights")
print("  - Number of Diagnoses:")
print(f"    Claims with overpayment ('I') have an average of {avg_num_diags_I:.2f} diagnoses.")
print(f"    Claims with no findings ('Z') have an average of {avg_num_diags_Z:.2f} diagnoses.")
if avg_num_diags_I > avg_num_diags_Z:
    print("    Insight: Claims with a higher number of diagnoses appear to have a higher overpayment risk.")
else:
    print("    Insight: The number of diagnoses does not seem to be a major differentiator for overpayment.")

print("\n  - Clinical Severity (Max SOI):")
print(f"    Claims with overpayment ('I') have an average max clinical SOI of {avg_max_soi_I:.2f}.")
print(f"    Claims with no findings ('Z') have an average max clinical SOI of {avg_max_soi_Z:.2f}.")
if avg_max_soi_I > avg_max_soi_Z:
    print("    Insight: Higher clinical severity seems to be associated with an increased likelihood of overpayment.")
else:
    print("    Insight: Clinical severity does not appear to be a significant factor in overpayment.")


# --- Insights from Billed Data ('aprdrg') ---
drg_overpayment_rates = df.groupby('aprdrg_drg')['sta'].apply(
    lambda x: (x == 'I').mean() * 100
).sort_values(ascending=False).reset_index(name='overpayment_rate')

top_3_drgs = drg_overpayment_rates.head(3)
drg_soi_rates = df.groupby('aprdrg_soi')['sta'].apply(
    lambda x: (x == 'I').mean() * 100
).reset_index(name='overpayment_rate')

print("\n### 3. Billed Data (APRDRG) Insights")
print("  - Top 3 APRDRG Codes with the highest overpayment rate:")
for index, row in top_3_drgs.iterrows():
    print(f"    - DRG '{row['aprdrg_drg']}' has an overpayment rate of {row['overpayment_rate']:.2f}%.")

print("\n  - Billed Severity (APRDRG SOI):")
for index, row in drg_soi_rates.iterrows():
    print(f"    - Claims with APRDRG SOI of {row['aprdrg_soi']} have an overpayment rate of {row['overpayment_rate']:.2f}%.")
if drg_soi_rates['overpayment_rate'].is_monotonic_increasing:
    print("    Insight: There is a strong positive correlation between billed severity and the overpayment rate.")


# --- Insights from Discrepancy Analysis ---
discrepancy_rate_I = df[df['soi_discrepancy']]['sta'].apply(lambda x: (x == 'I').mean()) * 100
no_discrepancy_rate_I = df[~df['soi_discrepancy']]['sta'].apply(lambda x: (x == 'I').mean()) * 100

print("\n### 4. Critical Discrepancy Insight")
print(f"  - Overpayment rate for claims WITH a clinical vs. billed SOI discrepancy: {discrepancy_rate_I:.2f}%")
print(f"  - Overpayment rate for claims WITHOUT a discrepancy: {no_discrepancy_rate_I:.2f}%")
if discrepancy_rate_I > no_discrepancy_rate_I:
    print("    Key Finding: Claims where the billed severity (APRDRG SOI) does not match the max clinical severity are significantly more likely to be overpaid. This is a powerful signal for potential errors or fraud.")
else:
    print("    Key Finding: Discrepancies between billed and clinical SOI do not seem to be a strong indicator of overpayment in this dataset.")

print("\n" + "="*80)
