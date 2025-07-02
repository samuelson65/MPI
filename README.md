import pandas as pd
import numpy as np # For np.nan

# Assuming drg_df_weights and drg_weight_map are available globally or passed
# For demonstration, let's define them again here:
drg_data = {
    'DRG_Code': ["DRG_481", "DRG_482", "DRG_291", "DRG_101", "DRG_999", "DRG_500", "DRG_600",
                 "DRG_NoCodes", "DRG_DiagOnly_D4_D5_D6", "DRG_Other_2_AB", "DRG_Other_1_A",
                 "DRG_Other_1_C", "DRG_Other_0_NoP", "DRG_700"], # Added DRG_700
    'Weight': [2.4819, 1.8000, 1.5000, 0.8000, 0.5000, 3.5000, 1.2000, 0.1000, 0.7000, 2.0000, 1.0000, 0.9000, 0.3000, 1.3000]
}
drg_df_weights = pd.DataFrame(drg_data)
drg_weight_map = drg_df_weights.set_index('DRG_Code')['Weight'].to_dict()

def get_drg_weight(drg_code, weight_map):
    """Safely gets the weight for a DRG code from the map. Returns inf if not found."""
    return weight_map.get(str(drg_code), float('inf'))

def generate_nlg_summary_from_lower_weight_df(df, drg_weight_map):
    """
    Generates a natural language summary suggesting potential DRG shifts
    based on the 'lower_weight_drgs_after_removal' column and frequency.

    Args:
        df (pd.DataFrame): The DataFrame with 'lower_weight_drgs_after_removal'
                           and 'billed_drg' columns.
        drg_weight_map (dict): A dictionary mapping DRG codes to their weights.

    Returns:
        str: A natural language summary.
    """
    summary_parts = ["--- DRG Cost Optimization Opportunities Analysis ---"]

    # 1. Overall potential shifts identified
    total_potential_shifts_rows = df['lower_weight_drgs_after_removal'].notna().sum()

    if total_potential_shifts_rows == 0:
        summary_parts.append("\nNo instances were identified where removing a single procedure could lead to a lower-weighted DRG.")
        summary_parts.append("This suggests that current procedure coding appears optimized for DRG assignment, or that alternative analyses (e.g., impact of multiple procedure changes) may be beneficial.")
        return "\n".join(summary_parts)

    summary_parts.append(
        f"\nOut of {len(df)} patient encounters reviewed, **{total_potential_shifts_rows} cases** present potential opportunities where a single procedure removal could result in a lower-weighted DRG."
    )
    summary_parts.append(
        "This indicates areas for focused review to ensure optimal DRG assignment and efficient resource utilization."
    )

    # 2. Prepare data for frequency analysis
    shift_analysis = []
    for index, row in df.iterrows():
        potential_shifts = row['lower_weight_drgs_after_removal']
        original_billed_drg = row['billed_drg']

        if pd.notna(potential_shifts) and isinstance(potential_shifts, dict):
            for removed_proc, new_drg in potential_shifts.items():
                shift_analysis.append({
                    'original_drg': original_billed_drg,
                    'removed_proc': removed_proc,
                    'new_drg': new_drg,
                    'original_weight': get_drg_weight(original_billed_drg, drg_weight_map),
                    'new_weight': get_drg_weight(new_drg, drg_weight_map)
                })

    if not shift_analysis: # Safety check, though unlikely if total_potential_shifts_rows > 0
        summary_parts.append("\nNo specific lower-weighted DRG shifts were detailed, despite identifying potential opportunities.")
        return "\n".join(summary_parts)

    shift_df = pd.DataFrame(shift_analysis)

    # Calculate weight difference for each potential shift
    shift_df['weight_difference'] = shift_df['original_weight'] - shift_df['new_weight']

    # Group by the proposed shift combination
    grouped_shifts = shift_df.groupby(['original_drg', 'removed_proc', 'new_drg']).agg(
        frequency=('original_drg', 'size'),
        total_weight_reduction=('weight_difference', 'sum') # Sum of weight reduction for this type of shift
    ).reset_index()

    # Sort by frequency (most common opportunities first)
    # Then by total weight reduction (most financially impactful within frequency)
    grouped_shifts = grouped_shifts.sort_values(
        by=['frequency', 'total_weight_reduction'],
        ascending=[False, False]
    )

    summary_parts.append("\nKey DRG shift patterns and their impact:")

    # Display top X most frequent/impactful shifts
    top_shifts_to_display = 5
    for idx, row in grouped_shifts.head(top_shifts_to_display).iterrows():
        original_drg = row['original_drg']
        removed_proc = row['removed_proc']
        new_drg = row['new_drg']
        frequency = row['frequency']
        total_weight_reduction = row['total_weight_reduction']

        # To get an average weight difference for clearer communication
        avg_weight_diff_per_case = total_weight_reduction / frequency

        summary_parts.append(
            f"- **Scenario: Billed DRG `{original_drg}`.** In **{frequency} cases**, removing procedure `{removed_proc}` could lead to a shift to DRG `{new_drg}`. This specific shift averages a weight reduction of **{avg_weight_diff_per_case:.3f}** per case."
        )
        if frequency > 1:
            summary_parts.append(f"  This is a recurring pattern indicating a significant area for review.")

    # 3. General Recommendations
    summary_parts.append("\n**Recommendations for Consideration:**")
    summary_parts.append(
        "1. **Targeted Review:** Prioritize clinical and coding review for the specific patient encounters and procedure-DRG shift combinations highlighted above. Understanding the clinical context of these procedures is crucial."
    )
    summary_parts.append(
        "2. **Documentation Best Practices:** Reinforce the importance of precise and comprehensive clinical documentation for all procedures performed. This ensures that the documentation accurately reflects the patient's condition and the services provided, supporting the most appropriate DRG assignment."
    )
    summary_parts.append(
        "3. **Coder Education:** Provide ongoing education to coders regarding the impact of specific procedure codes on DRG assignment, particularly for those procedures frequently leading to lower-weighted shifts upon removal."
    )
    summary_parts.append(
        "4. **Impact on Reimbursement vs. Accuracy:** While shifting to a lower-weighted DRG may impact reimbursement, the primary goal should be to ensure the most accurate DRG assignment that reflects the patient's severity of illness and resource consumption. This analysis aids in identifying areas where current coding might inadvertently lead to higher-than-justified DRGs."
    )

    return "\n".join(summary_parts)

# --- Example Usage (Assuming you have a DataFrame structured like this) ---

# This simulates the DataFrame you would pass to the function
# (after running the previous steps to populate 'lower_weight_drgs_after_removal')
example_df = pd.DataFrame({
    'patient_id': [101, 102, 103, 104, 105, 106, 107],
    'billed_drg': ['DRG_481', 'DRG_481', 'DRG_500', 'DRG_600', 'DRG_481', 'DRG_291', 'DRG_700'],
    'lower_weight_drgs_after_removal': [
        {'B': 'DRG_482', 'C': 'DRG_101'},  # Patient 101: B removal to DRG_482, C removal to DRG_101
        {'B': 'DRG_482'},                 # Patient 102: B removal to DRG_482
        np.nan,                           # Patient 103: No lower weight shifts
        {'Z': 'DRG_101'},                 # Patient 104: Z removal to DRG_101
        {'B': 'DRG_482', 'C': 'DRG_101'},  # Patient 105: B removal to DRG_482, C removal to DRG_101
        np.nan,                           # Patient 106: No lower weight shifts
        {'P': 'DRG_999'}                  # Patient 107: P removal to DRG_999
    ]
})

print("Input DataFrame for Summary Generation:")
print(example_df)
print("\n" + "="*50 + "\n")

# Generate the summary
summary_report = generate_nlg_summary_from_lower_weight_df(example_df, drg_weight_map)
print(summary_report)

