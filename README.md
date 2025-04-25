import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import shap
import plotly.graph_objects as go
from scipy.stats import ttest_1samp, spearmanr

# Sample data
data = pd.DataFrame({
    'DRG_Code': ['039', '039', '470', '470', '039', '470', '039', '470', '039', '470', '039', '470'],
    'Diagnosis_Code': ['I10', 'E11', 'I10', 'E11', 'I25', 'I25', 'I10', 'E11', 'I10', 'E11', 'I25', 'I25'],
    'Length_of_Stay': [3, 2, 5, 1, 4, 2, 6, 3, 2, 1, 5, 4],
    'Age': [65, 70, 60, 50, 80, 75, 68, 55, 72, 58, 66, 62],
    'Num_Procedures': [2, 1, 3, 1, 2, 2, 4, 1, 3, 1, 2, 3],
    'Overpayment_Flag': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]
})

# Encode categorical features
data['DRG_Code_enc'] = data['DRG_Code'].astype('category').cat.codes
data['Diagnosis_Code_enc'] = data['Diagnosis_Code'].astype('category').cat.codes

feature_cols = ['DRG_Code_enc', 'Diagnosis_Code_enc', 'Length_of_Stay', 'Age', 'Num_Procedures']
X = data[feature_cols]
y = data['Overpayment_Flag']

# Train model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X, y)
explainer = shap.TreeExplainer(model)

def generate_nlg_summary(drg_code, avg_risk, feature_stats, top_interactions):
    """Generate plain-language summary of key insights."""
    summary = [f"**Overpayment Risk Analysis for DRG {drg_code}**"]
    summary.append(f"- **Average predicted risk:** {avg_risk:.2f}/1.00")
    
    # Top risk drivers
    top_risk = feature_stats.nlargest(3, 'Mean_Abs_SHAP')
    summary.append("\n**Top Risk Drivers:**")
    for _, row in top_risk.iterrows():
        direction = "increases" if row['Mean_SHAP'] > 0 else "decreases"
        significance = "(statistically significant)" if row['Significant'] else "(not statistically significant)"
        summary.append(f"  - **{row['Feature']}**: {direction} risk {significance}")
    
    # Interactions
    summary.append("\n**Key Interactions:**")
    for inter in top_interactions[:2]:
        summary.append(f"  - **{inter['Feature1']}** and **{inter['Feature2']}** jointly influence risk")
    
    # Clinical flags
    if 'Length_of_Stay' in top_risk['Feature'].values:
        summary.append("\n**Clinical Flag:** Longer stays correlate with higher overpayment risk.")
    
    return "\n".join(summary)

def generate_queries(drg_code, top_features, top_interactions, df):
    """Generate conditional queries based on feature thresholds."""
    queries = []
    
    # Get median thresholds for numerical features
    los_median = df[df['DRG_Code'] == drg_code]['Length_of_Stay'].median()
    procedure_median = df[df['DRG_Code'] == drg_code]['Num_Procedures'].median()
    
    # Generate conditional queries for top features
    for feature in top_features['Feature'].head(3):
        if feature == 'Length_of_Stay':
            direction = "greater" if top_features.loc[top_features['Feature'] == feature, 'Mean_SHAP'].values[0] > 0 else "less"
            queries.append(
                f"Select DRG {drg_code} when length of stay is {direction} than {los_median} days"
            )
        elif feature == 'Diagnosis_Code_enc':
            common_diags = df[df['DRG_Code'] == drg_code]['Diagnosis_Code'].value_counts().index[:2]
            queries.append(
                f"Select DRG {drg_code} when diagnosis codes include {', '.join(common_diags)}"
            )
        elif feature == 'Num_Procedures':
            direction = "above" if top_features.loc[top_features['Feature'] == feature, 'Mean_SHAP'].values[0] > 0 else "below"
            queries.append(
                f"Select DRG {drg_code} when procedure count is {direction} {procedure_median}"
            )
    
    # Add interaction-based queries
    for inter in top_interactions[:2]:
        f1, f2 = inter['Feature1'], inter['Feature2']
        if 'Length_of_Stay' in {f1, f2} and 'Diagnosis_Code_enc' in {f1, f2}:
            common_diags = df[df['DRG_Code'] == drg_code]['Diagnosis_Code'].value_counts().index[:2]
            los_direction = "greater" if top_features.loc[top_features['Feature'] == 'Length_of_Stay', 'Mean_SHAP'].values[0] > 0 else "less"
            queries.append(
                f"Select DRG {drg_code} when length of stay is {los_direction} than {los_median} days "
                f"AND has diagnosis codes {', '.join(common_diags)}"
            )
    
    return queries

def explain_overpayment_for_drg(drg_code):
    drg_data = data[data['DRG_Code'] == drg_code]
    if drg_data.empty:
        return f"No data found for DRG {drg_code}", None, None
    
    X_drg = drg_data[feature_cols]
    y_drg = drg_data['Overpayment_Flag']
    
    # SHAP values
    shap_values = explainer.shap_values(X_drg)[1]
    feature_stats = pd.DataFrame({
        'Feature': feature_cols,
        'Mean_SHAP': np.mean(shap_values, axis=0),
        'Mean_Abs_SHAP': np.mean(np.abs(shap_values), axis=0)
    })
    
    # Permutation importance
    perm_imp = permutation_importance(model, X_drg, y_drg, n_repeats=100, random_state=42)
    feature_stats['Permutation_Importance'] = perm_imp['importances_mean']
    feature_stats['p_value'] = [ttest_1samp(perm_imp['importances'][i], 0).pvalue for i in range(len(feature_cols))]
    feature_stats['Significant'] = feature_stats['p_value'] < 0.05
    
    # Interactions
    shap_interaction = explainer.shap_interaction_values(X_drg)[1]
    interaction_means = np.mean(np.abs(shap_interaction), axis=0)
    interactions = []
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            interactions.append({
                'Feature1': feature_cols[i],
                'Feature2': feature_cols[j],
                'Mean_Abs_Interaction': interaction_means[i, j]
            })
    top_interactions = sorted(interactions, key=lambda x: x['Mean_Abs_Interaction'], reverse=True)
    
    # Generate outputs
    avg_risk = np.mean(model.predict_proba(X_drg)[:, 1])
    nlg_summary = generate_nlg_summary(drg_code, avg_risk, feature_stats, top_interactions)
    queries = generate_queries(drg_code, feature_stats, top_interactions, data)
    
    # Visualization
    first_shap = shap_values[0]
    influence_df = pd.DataFrame({
        'Feature': feature_cols,
        'SHAP_Value': first_shap
    }).sort_values(by='SHAP_Value', key=abs, ascending=False).head(5)
    
    fig = go.Figure(go.Bar(
        x=influence_df['SHAP_Value'],
        y=influence_df['Feature'],
        orientation='h',
        marker_color=['red' if x > 0 else 'blue' for x in influence_df['SHAP_Value']]
    ))
    fig.update_layout(
        title=f"Top Features Influencing Overpayment Risk (DRG {drg_code})",
        template='plotly_white'
    )
    
    return (nlg_summary, fig, queries)

# Example usage
result = explain_overpayment_for_drg('039')
if isinstance(result, tuple):
    summary, plot, queries = result
    print(summary)
    plot.show()
    print("\n**Suggested Queries:**")
    for i, q in enumerate(queries, 1):
        print(f"{i}. {q}")
else:
    print(result)
