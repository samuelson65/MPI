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

    print(f"\n=== Overpayment Risk Explanation for DRG {drg_code} ===")
    print(f"Number of claims analyzed: {len(drg_data)}")
    print(f"Average predicted overpayment risk: {probs.mean():.2f}\n")

    print("Top features influencing overpayment risk (with statistical significance):")
    for i, row in importance_df.head(7).iterrows():
        signif = "✅ Statistically significant" if row['p_value'] < 0.05 else "⚠️ Not statistically significant"

        # For categorical features, show example category names
        feature_name = row['Feature']
        if feature_name == 'Diagnosis_Code_enc':
            example_codes = drg_data['Diagnosis_Code'].unique()
            feature_desc = f"{feature_name} (e.g., {', '.join(example_codes)})"
        elif feature_name == 'DRG_Code_enc':
            feature_desc = f"{feature_name} (DRG {drg_code})"
        else:
            feature_desc = feature_name

        print(f"- {feature_desc}: mean SHAP={row['Mean_Abs_SHAP']:.3f}, "
              f"perm importance={row['Permutation_Importance']:.3f}, p-value={row['p_value']:.3f} {signif}")

    # Detailed SHAP values for the first claim in this DRG
    first_shap = shap_values[0]
    first_features = X_drg.iloc[0]

    influence_df = pd.DataFrame({
        'Feature': feature_cols,
        'SHAP_Value': first_shap,
        'Feature_Value': first_features.values
    }).sort_values(by='SHAP_Value', key=abs, ascending=False).head(10)

    # Color bars: red for positive impact (increases risk), blue for negative (decreases risk)
    colors = ['red' if val > 0 else 'blue' for val in influence_df['SHAP_Value']]

    # Replace encoded categorical values with readable labels for display
    def decode_feature_value(feature, val):
        if feature == 'Diagnosis_Code_enc':
            return diag_cat_map.get(val, val)
        elif feature == 'DRG_Code_enc':
            return drg_cat_map.get(val, val)
        else:
            return val

    influence_df['Feature_Value_Display'] = [
        decode_feature_value(f, v) for f, v in zip(influence_df['Feature'], influence_df['Feature_Value'])
    ]

    fig = go.Figure(go.Bar(
        x=influence_df['SHAP_Value'],
        y=influence_df['Feature'],
        orientation='h',
        marker_color=colors,
        text=influence_df['Feature_Value_Display'],
        textposition='auto'
    ))

    fig.update_layout(
        title=f"Top Feature Impacts on Overpayment Risk for First Claim in DRG {drg_code}",
        xaxis_title="SHAP Value (Impact on Prediction)",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed"),
        template='plotly_white'
    )

    fig.show()

# Example usage
explain_overpayment_for_drg('039')
