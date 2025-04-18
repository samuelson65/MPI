import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import shap
import plotly.graph_objects as go
from scipy.stats import ttest_1samp, spearmanr

# Sample data (same as before)
data = pd.DataFrame({
    'DRG_Code': ['039', '039', '470', '470', '039', '470', '039', '470', '039', '470', '039', '470'],
    'Diagnosis_Code': ['I10', 'E11', 'I10', 'E11', 'I25', 'I25', 'I10', 'E11', 'I10', 'E11', 'I25', 'I25'],
    'Length_of_Stay': [3, 2, 5, 1, 4, 2, 6, 3, 2, 1, 5, 4],
    'Age': [65, 70, 60, 50, 80, 75, 68, 55, 72, 58, 66, 62],
    'Num_Procedures': [2, 1, 3, 1, 2, 2, 4, 1, 3, 1, 2, 3],
    'Overpayment_Flag': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]
})

# Encoding categorical variables
data['DRG_Code_enc'] = data['DRG_Code'].astype('category').cat.codes
data['Diagnosis_Code_enc'] = data['Diagnosis_Code'].astype('category').cat.codes

feature_cols = ['DRG_Code_enc', 'Diagnosis_Code_enc', 'Length_of_Stay', 'Age', 'Num_Procedures']
X = data[feature_cols]
y = data['Overpayment_Flag']

model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X, y)

explainer = shap.TreeExplainer(model)

def bootstrap_p_values(model, X, y, n_bootstrap=100):
    result = permutation_importance(model, X, y, n_repeats=n_bootstrap, random_state=42)
    importances = result.importances_mean
    p_values = []
    for i in range(len(feature_cols)):
        _, p_val = ttest_1samp(result.importances[i], 0)
        p_values.append(p_val)
    return importances, p_values

def generate_natural_language_summary(feature_stats, drg_code, avg_risk):
    lines = [f"Overpayment risk analysis for DRG {drg_code}:"]
    lines.append(f"Average predicted risk is {avg_risk:.2f}.")
    for _, row in feature_stats.iterrows():
        direction = "increases" if row['Mean_SHAP'] > 0 else "decreases"
        signif = "significantly " if row['Significant'] else ""
        lines.append(
            f"- Feature '{row['Feature']}' {signif}{direction} the overpayment risk "
            f"with an average impact magnitude of {abs(row['Mean_SHAP']):.3f}."
        )
    return "\n".join(lines)

def explain_overpayment_for_drg(drg_code):
    drg_data = data[data['DRG_Code'] == drg_code]
    if drg_data.empty:
        print(f"No data for DRG {drg_code}")
        return

    X_drg = drg_data[feature_cols]
    y_drg = drg_data['Overpayment_Flag']

    probs = model.predict_proba(X_drg)[:, 1]
    shap_values = explainer.shap_values(X_drg)[1]

    # Feature-wise SHAP stats
    feature_stats = pd.DataFrame({
        'Feature': feature_cols,
        'Mean_SHAP': np.mean(shap_values, axis=0),
        'Median_SHAP': np.median(shap_values, axis=0),
        'Std_SHAP': np.std(shap_values, axis=0),
        'Mean_Abs_SHAP': np.mean(np.abs(shap_values), axis=0)
    })

    # Permutation importance with p-values
    perm_imp, p_vals = bootstrap_p_values(model, X_drg, y_drg)
    feature_stats['Permutation_Importance'] = perm_imp
    feature_stats['p_value'] = p_vals
    feature_stats['Significant'] = feature_stats['p_value'] < 0.05

    # Correlation with target
    corrs = []
    for f in feature_cols:
        corr, _ = spearmanr(drg_data[f], y_drg)
        corrs.append(corr)
    feature_stats['SpearmanCorr'] = corrs

    # Interaction effects (top 2 features only for simplicity)
    shap_interaction_values = explainer.shap_interaction_values(X_drg)[1]
    # Calculate mean absolute interaction for each feature pair
    interaction_means = np.mean(np.abs(shap_interaction_values), axis=0)
    # We'll just print top interactions for first two features for brevity
    top_interactions = []
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            top_interactions.append({
                'Feature1': feature_cols[i],
                'Feature2': feature_cols[j],
                'Mean_Abs_Interaction': interaction_means[i, j]
            })
    top_interactions = sorted(top_interactions, key=lambda x: x['Mean_Abs_Interaction'], reverse=True)[:3]

    # Natural language summary
    summary = generate_natural_language_summary(feature_stats, drg_code, probs.mean())
    print(summary)

    print("\nDetailed Feature Statistics:")
    print(feature_stats[['Feature', 'Mean_SHAP', 'Median_SHAP', 'Std_SHAP', 'Mean_Abs_SHAP', 'Permutation_Importance', 'p_value', 'Significant', 'SpearmanCorr']])

    print("\nTop Feature Interaction Effects:")
    for inter in top_interactions:
        print(f"- Interaction between '{inter['Feature1']}' and '{inter['Feature2']}': mean absolute impact {inter['Mean_Abs_Interaction']:.4f}")

    # Visualize top 7 features by mean absolute SHAP for first claim
    first_shap = shap_values[0]
    first_features = X_drg.iloc[0]
    influence_df = pd.DataFrame({
        'Feature': feature_cols,
        'SHAP_Value': first_shap,
        'Feature_Value': first_features.values
    }).sort_values(by='SHAP_Value', key=abs, ascending=False).head(7)

    colors = ['red' if val > 0 else 'blue' for val in influence_df['SHAP_Value']]

    # Decode categorical features for display
    diag_map = dict(enumerate(data['Diagnosis_Code'].astype('category').cat.categories))
    drg_map = dict(enumerate(data['DRG_Code'].astype('category').cat.categories))

    def decode(f, v):
        if f == 'Diagnosis_Code_enc':
            return diag_map.get(v, v)
        elif f == 'DRG_Code_enc':
            return drg_map.get(v, v)
        else:
            return v

    influence_df['Feature_Value_Display'] = [decode(f, v) for f, v in zip(influence_df['Feature'], influence_df['Feature_Value'])]

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
