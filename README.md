def generate_recommendations(patient_shap, patient_data, threshold=0.05):
    recommendations = []

    for feature, impact in patient_shap.items():
        if impact > threshold:  # Risk Factors
            if feature == "Length_of_Stay":
                recommendations.append("Consider early discharge planning with home care support to reduce length of stay risks.")
            elif feature == "Comorbidity_Index":
                recommendations.append("Enroll the patient in a post-discharge chronic disease management program.")
            elif feature == "Follow_Up_Days" and patient_data["Follow_Up_Days"] > 7:
                recommendations.append("Ensure follow-up visits occur within 7 days post-discharge to minimize risk.")
            elif feature == "Discharge_Disposition" and patient_data["Discharge_Disposition"] == "Home":
                recommendations.append("Schedule a nurse home visit within 48 hours after discharge.")
            elif feature == "Previous_Readmissions":
                recommendations.append("Assign a case manager for long-term follow-up.")

        elif impact < -threshold:  # Protective Factors
            if feature == "Early Follow-Up":
                recommendations.append("Continue timely follow-up appointments as they significantly reduce readmission risk.")
            elif feature == "Primary Care Engagement":
                recommendations.append("Encourage ongoing primary care visits to maintain stability.")
            elif feature == "Medication Adherence Score":
                recommendations.append("Maintain high medication adherence; consider patient education programs.")
            elif feature == "Rehabilitation Program":
                recommendations.append("Continue rehab program participation for better recovery outcomes.")

    return recommendations

# Example for one patient
patient_index = 0
patient_shap_values = shap_df.iloc[patient_index]
patient_data = X_test.iloc[patient_index]

recommendations = generate_recommendations(patient_shap_values, patient_data)
print(recommendations)




import pandas as pd
import numpy as np

# Convert SHAP values to DataFrame
shap_df = pd.DataFrame(shap_values, columns=X_test.columns)

# Compute mean SHAP values
mean_shap = shap_df.mean().sort_values(ascending=False)

# Separate positive (risk factors) and negative (protective factors)
positive_influences = mean_shap[mean_shap > 0]  # Features that increase readmission risk
negative_influences = mean_shap[mean_shap < 0]  # Features that reduce readmission risk

print("Risk Factors (Increase Readmission):")
print(positive_influences.head(10))

print("\nProtective Factors (Reduce Readmission):")
print(negative_influences.tail(10))



def get_drg_weight(drg_df, drg_code, date_of_service):
    """
    Given a DataFrame with DRG codes, their weights, and effective/term dates, 
    this function returns the weight for a given DRG code on a specific date of service.
    
    Parameters:
    drg_df (DataFrame): DataFrame containing 'drg', 'weight', 'effective_date', and 'term_date'.
    drg_code (int or str): The DRG code to look up.
    date_of_service (datetime or str): The service date to check.

    Returns:
    float or None: The weight of the DRG if found, else None.
    """
    # Ensure date_of_service is a datetime object
    date_of_service = pd.to_datetime(date_of_service)

    # Filter for the DRG code and date range
    filtered_df = drg_df[
        (drg_df['drg'] == drg_code) &
        (drg_df['effective_date'] <= date_of_service) &
        (drg_df['term_date'] >= date_of_service)
    ]

    # Return the weight if a match is found
    if not filtered_df.empty:
        return filtered_df.iloc[0]['weight']  # Assuming one valid record per DRG-date range
    else:
        return None  # No matching DRG for the given date

def get_drg_weight(drg_df, drg_code, date_of_service):
    """
    Fetch the DRG weight based on the given DRG code and date of service.
    Returns None if the DRG code is not found or out of the valid date range.
    """
    date_of_service = pd.to_datetime(date_of_service)

    # Ensure 'drg' column exists in drg_df
    if 'drg' not in drg_df.columns:
        raise KeyError("'drg' column is missing from the DRG DataFrame")

    # Filter DRG codes that match and fall within the date range
    filtered_df = drg_df[
        (drg_df['drg'] == drg_code) & 
        (drg_df['effective_date'] <= date_of_service) & 
        (drg_df['term_date'] >= date_of_service)
    ]

    # If no valid DRG entry is found, return None
    if filtered_df.empty:
        return None

    return filtered_df.iloc[0]['weight']


def filter_lower_weight_drgs(row, drg_df):
    date_of_service = row.get('date_of_service', None)  # Get date_of_service safely
    if date_of_service is None:
        print(f"Missing date_of_service in row: {row}")
        return {}

    actual_drg_weight = get_drg_weight(drg_df, row['actual_drg'], date_of_service)
    
    if actual_drg_weight is None:
        print(f"DRG weight not found for actual DRG {row['actual_drg']} on {date_of_service}")
        return {}

    drg_map = row.get('result', {})  # Get 'result' safely, default to empty dict

    filtered_drg_map = {}
    
    for diag, drg_list in drg_map.items():
        if not isinstance(drg_list, list):
            print(f"Unexpected non-list value for DRG mapping in {diag}: {drg_list}")
            continue

        filtered_drg_map[diag] = [
            drg for drg in drg_list 
            if (drg_weight := get_drg_weight(drg_df, drg, date_of_service)) is not None 
            and drg_weight < actual_drg_weight
        ]

    return filtered_drg_map

# Apply function to DataFrame
df['filtered_result'] = df.apply(lambda row: filter_lower_weight_drgs(row, drg_df), axis=1)
