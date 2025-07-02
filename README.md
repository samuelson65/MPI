import pandas as pd
import numpy as np
from datetime import datetime # To include current date/time for reporting

# Assume drg_df_weights and drg_weight_map are available globally or passed
# For demonstration, let's define them again here:
drg_data = {
    'DRG_Code': ["DRG_481", "DRG_482", "DRG_291", "DRG_101", "DRG_999", "DRG_500", "DRG_600",
                 "DRG_NoCodes", "DRG_DiagOnly_D4_D5_D6", "DRG_Other_2_AB", "DRG_Other_1_A",
                 "DRG_Other_1_C", "DRG_Other_0_NoP", "DRG_700"],
    'Weight': [2.4819, 1.8000, 1.5000, 0.8000, 0.5000, 3.5000, 1.2000, 0.1000, 0.7000, 2.0000, 1.0000, 0.9000, 0.3000, 1.3000]
}
drg_df_weights = pd.DataFrame(drg_data)
drg_weight_map = drg_df_weights.set_index('DRG_Code')['Weight'].to_dict()

def get_drg_weight(drg_code, weight_map):
    """Safely gets the weight for a DRG code from the map. Returns inf if not found."""
    return weight_map.get(str(drg_code), float('inf'))

def generate_nlg_summary_from_lower_weight_df(df, drg_weight_map):
    """
    Generates a crisp and effective natural language summary of DRG shift opportunities.

    Args:
        df (pd.DataFrame): The DataFrame with 'lower_weight_drgs_after_removal'
                           and 'billed_drg' columns.
        drg_weight_map (dict): A dictionary mapping DRG codes to their weights.

    Returns:
        str: A concise natural language summary.
    """
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    summary_parts = [
        f"--- DRG Optimization Opportunities Analysis | Report Date: {report_date} ---",
        "\nThis report identifies potential DRG shifts resulting from the removal of specific procedure codes, highlighting opportunities for coding review and resource optimization."
    ]

    total_encounters = len(df)
    opportunities_count = df['lower_weight_drgs_after_removal'].notna().sum()

    if opportunities_count == 0:
        summary_parts.append(
            f"\nNo immediate DRG optimization opportunities were identified across {total_encounters} patient encounters based on single procedure removal. Current coding appears optimized, or further multi-procedure analysis may be beneficial."
        )
        return "\n".join(summary_parts)

    summary_parts.append(
        f"\n**Summary of Findings:**\n"
        f"Out of {total_encounters} patient encounters analyzed, **{opportunities_count} cases** ({opportunities_count/total_encounters:.1%}) present opportunities where removing a single procedure could lead to a lower-weighted DRG. This suggests areas for focused review to ensure optimal DRG assignment."
    )

    # Prepare data for frequency analysis
    shift_analysis = []
    for _, row in df.iterrows():
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

    shift_df = pd.DataFrame(shift_analysis)
    shift_df['weight_difference'] = shift_df['original_weight'] - shift_df['new_weight']

    # Group by the proposed shift combination
    grouped_shifts = shift_df.groupby(['original_drg', 'removed_proc', 'new_drg']).agg(
        frequency=('original_drg', 'size'),
        avg_weight_reduction=('weight_difference', 'mean') # Use mean for average impact
    ).reset_index()

    grouped_shifts = grouped_shifts.sort_values(
        by=['frequency', 'avg_weight_reduction'],
        ascending=[False, False]
    )

    summary_parts.append("\n**Key Shift Patterns:**")
    top_shifts_to_display = min(5, len(grouped_shifts)) # Display up to 5, or fewer if less exist

    for idx, row in grouped_shifts.head(top_shifts_to_display).iterrows():
        original_drg = row['original_drg']
        removed_proc = row['removed_proc']
        new_drg = row['new_drg']
        frequency = row['frequency']
        avg_weight_diff = row['avg_weight_reduction']

        summary_parts.append(
            f"- **DRG `{original_drg}` → `{new_drg}`** (by removing `{removed_proc}`): Observed in **{frequency} cases** with an average weight reduction of **{avg_weight_diff:.3f}**."
        )

    # 3. Streamlined Recommendations
    summary_parts.append("\n**Recommendations:**")
    summary_parts.append(
        "1. **Prioritize Review:** Focus on cases matching the 'Key Shift Patterns' for clinical and coding review."
    )
    summary_parts.append(
        "2. **Enhance Documentation:** Ensure clear and comprehensive documentation supports all billed procedures and their necessity."
    )
    summary_parts.append(
        "3. **Refine Coding Practices:** Educate coders on the DRG impact of specific procedures and best practices for accurate assignment."
    )
    summary_parts.append(
        "4. **Outcome Focus:** Aim for the most accurate DRG, reflecting true patient severity and resource use, rather than solely focusing on reimbursement."
    )

    return "\n".join(summary_parts)

# --- Example Usage (Continued from previous steps) ---

# This simulates the DataFrame you would pass to the function
example_df = pd.DataFrame({
    'patient_id': [101, 102, 103, 104, 105, 106, 107],
    'billed_drg': ['DRG_481', 'DRG_481', 'DRG_500', 'DRG_600', 'DRG_481', 'DRG_291', 'DRG_700'],
    'lower_weight_drgs_after_removal': [
        {'B': 'DRG_482', 'C': 'DRG_101'},
        {'B': 'DRG_482'},
        np.nan,
        {'Z': 'DRG_101'},
        {'B': 'DRG_482', 'C': 'DRG_101'},
        np.nan,
        {'P': 'DRG_999'}
    ]
})

print("Input DataFrame for Summary Generation:")
print(example_df)
print("\n" + "="*50 + "\n")

# Generate the summary report string
summary_report_content = generate_nlg_summary_from_lower_weight_df(example_df, drg_weight_map)
print(summary_report_content)

# Optional: Write to file
# with open("DRG_Optimization_Summary_Report_Crisp.txt", "w") as f: f.write(summary_report_content)
