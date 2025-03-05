def generate_recommendations(patient_shap, patient_data, threshold=0.05):
    recommendations = []
    
    for feature, impact in patient_shap.items():
        if impact > threshold:  # Consider only high-impact factors
            if feature == "Length_of_Stay":
                recommendations.append("Consider a structured discharge plan, including home health visits.")
            elif feature == "Comorbidity_Index":
                recommendations.append("Enroll the patient in a transitional care program for closer monitoring.")
            elif feature == "Follow_Up_Days" and patient_data["Follow_Up_Days"] > 7:
                recommendations.append("Schedule a follow-up visit within 7 days post-discharge to reduce risk.")
            elif feature == "Discharge_Disposition" and patient_data["Discharge_Disposition"] == "Home":
                recommendations.append("Provide a nurse follow-up call within 48 hours to assess needs.")
            elif feature == "Previous_Readmissions" and patient_data["Previous_Readmissions"] > 1:
                recommendations.append("Flag for case management intervention due to multiple prior readmissions.")

    return recommendations

# Example for one patient
patient_index = 0
patient_shap_values = shap_df.iloc[patient_index]
patient_data = X_test.iloc[patient_index]

recommendations = generate_recommendations(patient_shap_values, patient_data)
print(recommendations)
