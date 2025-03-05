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
