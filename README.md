idef generate_recommendations(patient_shap, patient_data, threshold=0.05):
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


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
import plotly.graph_objects as go

# Sample data: claims with DRG, diagnosis, length of stay, age, procedures, and overpayment flag
data = pd.DataFrame({
    'DRG_Code': ['039', '039', '470', '470', '039', '470', '039', '470'],
    'Diagnosis_Code': ['I10', 'E11', 'I10', 'E11', 'I25', 'I25', 'I10', 'E11'],
    'Length_of_Stay': [3, 2, 5, 1, 4, 2, 6, 3],
    'Age': [65, 70, 60, 50, 80, 75, 68, 55],
    'Num_Procedures': [2, 1, 3, 1, 2, 2, 4, 1],
    'Overpayment_Flag': [1, 1, 0, 1, 0, 1, 1, 0]
})

# Encode categorical variables
data['DRG_Code_enc'] = data['DRG_Code'].astype('category').cat.codes
data['Diagnosis_Code_enc'] = data['Diagnosis_Code'].astype('category').cat.codes

# Features and target
feature_cols = ['DRG_Code_enc', 'Diagnosis_Code_enc', 'Length_of_Stay', 'Age', 'Num_Procedures']
X = data[feature_cols]
y = data['Overpayment_Flag']

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

def explain_overpayment_for_drg(drg_code):
    # Filter claims for the given DRG code
    drg_data = data[data['DRG_Code'] == drg_code]
    if drg_data.empty:
        print(f"No data for DRG code {drg_code}")
        return

    X_drg = drg_data[feature_cols]
    shap_values = explainer.shap_values(X_drg)[1]  # Class 1 (overpayment) SHAP values

    # Aggregate mean absolute SHAP values by feature for this DRG
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Abs_SHAP', ascending=False)

    # Also show direction of influence for the first claim (example)
    first_shap = shap_values[0]
    first_features = X_drg.iloc[0]

    influence_df = pd.DataFrame({
        'Feature': feature_cols,
        'SHAP_Value': first_shap,
        'Feature_Value': first_features.values
    }).sort_values(by='SHAP_Value', key=abs, ascending=False)

    print(f"\nTop features influencing overpayment risk for DRG {drg_code} (mean absolute SHAP):")
    print(feature_importance)

    print(f"\nSHAP values for first claim in DRG {drg_code}:")
    print(influence_df)

    # Plot top 5 features by absolute SHAP value for the first claim
    top_features = influence_df.head(5)
    colors = ['red' if val > 0 else 'blue' for val in top_features['SHAP_Value']]

    fig = go.Figure(go.Bar(
        x=top_features['SHAP_Value'],
        y=top_features['Feature'],
        orientation='h',
        marker_color=colors,
        text=top_features['Feature_Value'],
        textposition='auto'
    ))
    fig.update_layout(
        title=f"Top Influential Features for Overpayment Prediction (DRG {drg_code})",
        xaxis_title="SHAP Value (Impact on Prediction)",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed")
    )
    fig.show()

# Example usage: explain overpayment for DRG '039'
explain_overpayment_for_drg('039')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import shap
import plotly.graph_objects as go
from scipy.stats import ttest_1samp

# Sample data: claims with DRG, diagnosis, length of stay, age, procedures, and overpayment flag
data = pd.DataFrame({
    'DRG_Code': ['039', '039', '470', '470', '039', '470', '039', '470', '039', '470', '039', '470'],
    'Diagnosis_Code': ['I10', 'E11', 'I10', 'E11', 'I25', 'I25', 'I10', 'E11', 'I10', 'E11', 'I25', 'I25'],
    'Length_of_Stay': [3, 2, 5, 1, 4, 2, 6, 3, 2, 1, 5, 4],
    'Age': [65, 70, 60, 50, 80, 75, 68, 55, 72, 58, 66, 62],
    'Num_Procedures': [2, 1, 3, 1, 2, 2, 4, 1, 3, 1, 2, 3],
    'Overpayment_Flag': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]
})

# Encode categorical variables
data['DRG_Code_enc'] = data['DRG_Code'].astype('category').cat.codes
data['Diagnosis_Code_enc'] = data['Diagnosis_Code'].astype('category').cat.codes

# Features and target
feature_cols = ['DRG_Code_enc', 'Diagnosis_Code_enc', 'Length_of_Stay', 'Age', 'Num_Procedures']
X = data[feature_cols]
y = data['Overpayment_Flag']

# Train Random Forest model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X, y)

# SHAP explainer
explainer = shap.TreeExplainer(model)

def bootstrap_p_values(model, X, y, n_bootstrap=100):
    """
    Calculate p-values for permutation feature importance using bootstrapping.
    """
    result = permutation_importance(model, X, y, n_repeats=n_bootstrap, random_state=42)
    importances = result.importances_mean
    importances_std = result.importances_std

    # Null hypothesis: importance = 0
    # Use t-test across repeats for each feature
    p_values = []
    for i in range(len(feature_cols)):
        t_stat, p_val = ttest_1samp(result.importances[i], 0)
        p_values.append(p_val)
    return importances, p_values

def explain_overpayment_for_drg(drg_code):
    drg_data = data[data['DRG_Code'] == drg_code]
    if drg_data.empty:
        print(f"No data found for DRG code {drg_code}")
        return

    X_drg = drg_data[feature_cols]
    y_drg = drg_data['Overpayment_Flag']

    # Predict probabilities
    probs = model.predict_proba(X_drg)[:, 1]

    # SHAP values for DRG claims
    shap_values = explainer.shap_values(X_drg)[1]  # class 1 (overpayment)

    # Mean absolute SHAP values per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Abs_SHAP', ascending=False)

    # Permutation importance and p-values for statistical significance
    perm_importance, p_values = bootstrap_p_values(model, X_drg, y_drg)

    perm_df = pd.DataFrame({
        'Feature': feature_cols,
        'Permutation_Importance': perm_importance,
        'p_value': p_values
    }).sort_values(by='Permutation_Importance', ascending=False)

    # Merge SHAP and permutation importance info
    importance_df = pd.merge(shap_importance_df, perm_df, on='Feature')

    # Focus on Diagnosis_Code_enc for layman interpretation
    diag_code_idx = feature_cols.index('Diagnosis_Code_enc')
    diag_shap_mean = mean_abs_shap[diag_code_idx]
    diag_perm_imp = perm_importance[diag_code_idx]
    diag_p_val = p_values[diag_code_idx]

    # Interpretation text for diagnosis code
    diag_cat_map = dict(enumerate(data['Diagnosis_Code'].astype('category').cat.categories))
    diag_codes_in_drg = drg_data['Diagnosis_Code'].unique()

    print(f"\n=== Overpayment Risk Explanation for DRG {drg_code} ===")
    print(f"Number of claims analyzed: {len(drg_data)}")
    print(f"Average predicted overpayment risk: {probs.mean():.2f}")

    print("\nTop features influencing overpayment risk (by SHAP values):")
    for i, row in importance_df.head(5).iterrows():
        signif = " (statistically significant)" if row['p_value'] < 0.05 else " (not statistically significant)"
        print(f"- {row['Feature']}: mean SHAP={row['Mean_Abs_SHAP']:.3f}, "
              f"perm importance={row['Permutation_Importance']:.3f}, p-value={row['p_value']:.3f}{signif}")

    print("\nDiagnosis codes present in this DRG and their likely influence:")
    for code in diag_codes_in_drg:
        code_enc = data[data['Diagnosis_Code'] == code]['Diagnosis_Code_enc'].iloc[0]
        # Average SHAP for diagnosis code feature when this code is present
        mask = drg_data['Diagnosis_Code_enc'] == code_enc
        avg_shap = np.abs(shap_values[mask, diag_code_idx]).mean()
        print(f"  - Diagnosis code '{code}': average influence on risk = {avg_shap:.3f}")

    # Visualize top features for first claim in DRG
    first_shap = shap_values[0]
    first_features = X_drg.iloc[0]

    influence_df = pd.DataFrame({
        'Feature': feature_cols,
        'SHAP_Value': first_shap,
        'Feature_Value': first_features.values
    }).sort_values(by='SHAP_Value', key=abs, ascending=False).head(7)

    colors = ['red' if val > 0 else 'blue' for val in influence_df['SHAP_Value']]

    fig = go.Figure(go.Bar(
        x=influence_df['SHAP_Value'],
        y=influence_df['Feature'],
        orientation='h',
        marker_color=colors,
        text=influence_df['Feature_Value'],
        textposition='auto'
    ))
    fig.update_layout(
        title=f"Top Influential Features for Overpayment Prediction (DRG {drg_code}) - First Claim",
        xaxis_title="SHAP Value (Impact on Prediction)",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed")
    )
    fig.show()

# Example usage
explain_overpayment_for_drg('039')


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import shap
import plotly.graph_objects as go
from scipy.stats import ttest_1samp

# Sample data: claims with DRG, diagnosis, length of stay, age, procedures, and overpayment flag
data = pd.DataFrame({
    'DRG_Code': ['039', '039', '470', '470', '039', '470', '039', '470', '039', '470', '039', '470'],
    'Diagnosis_Code': ['I10', 'E11', 'I10', 'E11', 'I25', 'I25', 'I10', 'E11', 'I10', 'E11', 'I25', 'I25'],
    'Length_of_Stay': [3, 2, 5, 1, 4, 2, 6, 3, 2, 1, 5, 4],
    'Age': [65, 70, 60, 50, 80, 75, 68, 55, 72, 58, 66, 62],
    'Num_Procedures': [2, 1, 3, 1, 2, 2, 4, 1, 3, 1, 2, 3],
    'Overpayment_Flag': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]
})

# Encode categorical variables
data['DRG_Code_enc'] = data['DRG_Code'].astype('category').cat.codes
data['Diagnosis_Code_enc'] = data['Diagnosis_Code'].astype('category').cat.codes

# Features and target
feature_cols = ['DRG_Code_enc', 'Diagnosis_Code_enc', 'Length_of_Stay', 'Age', 'Num_Procedures']
X = data[feature_cols]
y = data['Overpayment_Flag']

# Train Random Forest model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X, y)

# SHAP explainer
explainer = shap.TreeExplainer(model)

def bootstrap_p_values(model, X, y, n_bootstrap=100):
    """
    Calculate p-values for permutation feature importance using bootstrapping.
    """
    result = permutation_importance(model, X, y, n_repeats=n_bootstrap, random_state=42)
    importances = result.importances_mean
    importances_std = result.importances_std

    # Null hypothesis: importance = 0
    # Use t-test across repeats for each feature
    p_values = []
    for i in range(len(feature_cols)):
        t_stat, p_val = ttest_1samp(result.importances[i], 0)
        p_values.append(p_val)
    return importances, p_values

def explain_overpayment_for_drg(drg_code):
    drg_data = data[data['DRG_Code'] == drg_code]
    if drg_data.empty:
        print(f"No data found for DRG code {drg_code}")
        return

    X_drg = drg_data[feature_cols]
    y_drg = drg_data['Overpayment_Flag']

    # Predict probabilities
    probs = model.predict_proba(X_drg)[:, 1]

    # SHAP values for DRG claims
    shap_values = explainer.shap_values(X_drg)[1]  # class 1 (overpayment)

    # Mean absolute SHAP values per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Abs_SHAP', ascending=False)

    # Permutation importance and p-values for statistical significance
    perm_importance, p_values = bootstrap_p_values(model, X_drg, y_drg)

    perm_df = pd.DataFrame({
        'Feature': feature_cols,
        'Permutation_Importance': perm_importance,
        'p_value': p_values
    }).sort_values(by='Permutation_Importance', ascending=False)

    # Merge SHAP and permutation importance info
    importance_df = pd.merge(shap_importance_df, perm_df, on='Feature')

    # Map categorical features back to original categories for interpretation
    diag_cat_map = dict(enumerate(data['Diagnosis_Code'].astype('category').cat.categories))
    drg_cat_map = dict(enumerate(data['DRG_Code'].astype('category').cat.categories))

    # Layman-friendly explanation for diagnosis and DRG codes
    diag_codes_in_drg = drg_data['Diagnosis_Code'].unique()
    drg_name = drg_code

    print(f"\n=== Overpayment Risk Explanation for DRG {drg_name} ===")
    print(f"Number of claims analyzed: {len(drg_data)}")
    print(f"Average predicted overpayment risk: {probs.mean():.2f}")

    print("\nTop features influencing overpayment risk (with statistical significance):")
    for i, row in importance_df.head(7).iterrows():
        signif = "✅ Statistically significant" if row['p_value'] < 0.05 else "⚠️ Not statistically significant"
        print(f"- {row['Feature']}: mean SHAP={row['Mean_Abs_SHAP']:.3f}, "
              f"perm importance={row['Permutation_Importance']:.3f}, p-value={row['p_value']:.3f} {signif}")

    print("\nDiagnosis codes present in this DRG and their average influence on risk:")
    diag_code_idx = feature_cols.index('Diagnosis_Code_enc')
    for code in diag_codes_in_drg:
        code_enc = data[data['Diagnosis_Code'] == code]['Diagnosis_Code_enc'].iloc[0]
        mask = drg_data['Diagnosis_Code_enc'] == code_enc
        avg_shap = np.abs(shap_values[mask, diag_code_idx]).mean()
        print(f"  - Diagnosis code '{code}': average SHAP influence = {avg_shap:.3f}")

    # Show detailed SHAP values for the first claim in this DRG
    first_shap = shap_values[0]
    first_features = X_drg.iloc[0]

    influence_df = pd.DataFrame({
        'Feature': feature_cols,
        'SHAP_Value': first_shap,
        'Feature_Value': first_features.values
    }).sort_values(by='SHAP_Value', key=abs, ascending=False).head(10)

    # Color bars: red for positive impact (increases risk), blue for negative (decreases risk)
    colors = ['red' if val > 0 else 'blue' for val in influence_df['SHAP_Value']]

    fig = go.Figure(go.Bar(
        x=influence_df['SHAP_Value'],
        y=influence_df['Feature'],
        orientation='h',
        marker_color=colors,
        text=influence_df['Feature_Value'],
        textposition='auto'
    ))

    fig.update_layout(
        title=f"Top Feature Impacts on Overpayment Risk for First Claim in DRG {drg_name}",
        xaxis_title="SHAP Value (Impact on Prediction)",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed"),
        template='plotly_white'
    )

    fig.show()

# Example usage
explain_overpayment_for_drg('039')


