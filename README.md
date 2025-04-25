import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import shap
import plotly.graph_objects as go
from scipy.stats import ttest_1samp

# Generate synthetic healthcare data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'DRG_Code': np.random.choice(['039', '470', '123', '456', '789', 'ABC', 'DEF', 'GHI'], size=n_samples),
    'Provider': np.random.choice(['HOSP_A', 'HOSP_B', 'HOSP_C', 'HOSP_D', 'HOSP_E'], size=n_samples),
    'MCC': np.random.choice(['MCC_1', 'MCC_2', 'MCC_3', 'MCC_4', 'MCC_5'], size=n_samples),
    'CC': np.random.choice(['CC_1', 'CC_2', 'CC_3', 'CC_4', 'None'], size=n_samples, p=[0.2, 0.2, 0.2, 0.2, 0.2]),
    'Diagnosis_Code': np.random.choice(
        ['I10', 'E11', 'I25', 'J18', 'N39', 'M54', 'G20', 'E78', 'K57'], 
        size=n_samples
    ),
    'Procedure_Codes': [
        ','.join(np.random.choice(
            ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'], 
            size=np.random.randint(1,5))
        ) for _ in range(n_samples)
    ],
    'Length_of_Stay': np.random.randint(1, 15, size=n_samples),
    'Age': np.random.randint(18, 95, size=n_samples),
    'Num_Procedures': np.random.randint(1, 6, size=n_samples),
    'Admission_Type': np.random.choice(
        ['Emergency', 'Urgent', 'Elective', 'Trauma', 'Newborn'], 
        size=n_samples
    ),
    'Discharge_Status': np.random.choice(
        ['Home', 'Facility', 'AMA', 'Hospice', 'Expired'], 
        size=n_samples
    ),
    'Payment_Type': np.random.choice(
        ['Medicare', 'Medicaid', 'Commercial', 'Self-Pay', 'Other'], 
        size=n_samples
    ),
    'Cost_Center': np.random.choice(
        ['CC_ICU', 'CC_OR', 'CC_LAB', 'CC_RAD', 'CC_PHARM'], 
        size=n_samples
    ),
    'Overpayment_Flag': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
})

# Feature engineering
for code in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']:
    data[f'Procedure_{code}'] = data['Procedure_Codes'].str.contains(code).astype(int)

# Encode categorical features
cat_cols = [
    'DRG_Code', 'Provider', 'MCC', 'CC', 'Diagnosis_Code', 
    'Admission_Type', 'Discharge_Status', 'Payment_Type', 'Cost_Center'
]
for col in cat_cols:
    data[f'{col}_enc'] = data[col].astype('category').cat.codes

# Feature columns
feature_cols = (
    [f'{col}_enc' for col in cat_cols] +
    ['Length_of_Stay', 'Age', 'Num_Procedures'] +
    [f'Procedure_{code}' for code in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']]
)

X = data[feature_cols]
y = data['Overpayment_Flag']

# Train model
model = RandomForestClassifier(random_state=42, n_estimators=150, max_depth=10)
model.fit(X, y)
explainer = shap.TreeExplainer(model)

def generate_nlg_summary(drg_code, avg_risk, feature_stats, top_interactions):
    summary = [f"**Overpayment Risk Analysis for DRG {drg_code}**"]
    summary.append(f"- **Average predicted risk:** {avg_risk:.2f}/1.00")
    
    # Top risk drivers
    top_risk = feature_stats.nlargest(5, 'Mean_Abs_SHAP')
    summary.append("\n**Top Risk Drivers:**")
    for _, row in top_risk.iterrows():
        direction = "increases" if row['Mean_SHAP'] > 0 else "decreases"
        significance = "(statistically significant)" if row['Significant'] else "(not statistically significant)"
        summary.append(f"  - **{row['Feature']}**: {direction} risk {significance}")
    
    # Interactions
    summary.append("\n**Key Interactions:**")
    for inter in top_interactions[:3]:
        summary.append(f"  - **{inter['Feature1']}** and **{inter['Feature2']}** jointly influence risk")
    
    # Special flags
    if 'Length_of_Stay' in top_risk['Feature'].values:
        summary.append("\n**Clinical Flag:** Longer stays correlate with higher overpayment risk.")
    if 'Procedure_P1' in top_risk['Feature'].values:
        summary.append("\n**Procedure Flag:** Claims with Procedure P1 show elevated risk.")
    
    return "\n".join(summary)

def generate_queries(drg_code, top_features, top_interactions, df):
    queries = []
    drg_data = df[df['DRG_Code'] == drg_code]
    
    # Common patterns
    common_providers = drg_data['Provider'].value_counts().index[:2]
    common_mcc = drg_data['MCC'].value_counts().index[0]
    common_cc = drg_data['CC'].value_counts().index[0]
    common_diags = drg_data['Diagnosis_Code'].value_counts().index[:2]
    common_payment = drg_data['Payment_Type'].value_counts().index[0]
    
    # Feature-specific queries
    for feature in top_features['Feature'].head(5):
        if feature == 'Length_of_Stay':
            los_median = drg_data['Length_of_Stay'].median()
            direction = "longer" if top_features.loc[top_features['Feature'] == feature, 'Mean_SHAP'].values[0] > 0 else "shorter"
            queries.append(f"Review DRG {drg_code} claims with {direction} than median LOS ({los_median} days)")
            
        elif feature == 'Provider_enc':
            queries.append(f"Audit {common_providers[0]} and {common_providers[1]} for DRG {drg_code} coding patterns")
            
        elif feature == 'MCC_enc':
            queries.append(f"Verify MCC {common_mcc} documentation for DRG {drg_code}")
            
        elif feature == 'Procedure_P1':
            queries.append(f"Investigate Procedure P1 utilization in DRG {drg_code}")
    
    # Interaction queries
    for inter in top_interactions[:2]:
        f1, f2 = inter['Feature1'], inter['Feature2']
        
        if 'Provider_enc' in {f1, f2} and 'MCC_enc' in {f1, f2}:
            queries.append(
                f"Examine DRG {drg_code} claims from {common_providers[0]} with MCC {common_mcc}"
            )
            
        if 'Payment_Type_enc' in {f1, f2} and 'Cost_Center_enc' in {f1, f2}:
            common_cost_center = drg_data['Cost_Center'].value_counts().index[0]
            queries.append(
                f"Analyze {common_payment} payments in {common_cost_center} for DRG {drg_code}"
            )
    
    return queries

def explain_overpayment_for_drg(drg_code):
    drg_data = data[data['DRG_Code'] == drg_code]
    if drg_data.empty:
        return f"No data found for DRG {drg_code}", None, None
    
    X_drg = drg_data[feature_cols]
    y_drg = drg_data['Overpayment_Flag']
    
    # SHAP analysis
    shap_values = explainer.shap_values(X_drg)[1]
    feature_stats = pd.DataFrame({
        'Feature': feature_cols,
        'Mean_SHAP': np.mean(shap_values, axis=0),
        'Mean_Abs_SHAP': np.mean(np.abs(shap_values), axis=0)
    })
    
    # Permutation importance
    perm_imp = permutation_importance(model, X_drg, y_drg, n_repeats=50, random_state=42)
    feature_stats['Permutation_Importance'] = perm_imp['importances_mean']
    feature_stats['p_value'] = [ttest_1samp(perm_imp['importances'][i], 0).pvalue for i in range(len(feature_cols))]
    feature_stats['Significant'] = feature_stats['p_value'] < 0.05
    
    # Interaction analysis
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
    
# ... [Previous code remains exactly the same until the visualization section]

    # Visualization
    influence_df = pd.DataFrame({
        'Feature': feature_cols,
        'SHAP_Value': np.mean(np.abs(shap_values), axis=0)
    }).sort_values('SHAP_Value', ascending=False).head(15)

    # Feature name mapping
    feature_map = {
        'DRG_Code_enc': 'DRG Code',
        'Provider_enc': 'Provider',
        'MCC_enc': 'MCC',
        'CC_enc': 'CC',
        'Diagnosis_Code_enc': 'Diagnosis',
        'Admission_Type_enc': 'Admission Type',
        'Discharge_Status_enc': 'Discharge Status',
        'Payment_Type_enc': 'Payment Type',
        'Cost_Center_enc': 'Cost Center',
        'Length_of_Stay': 'Length of Stay',
        'Age': 'Patient Age',
        'Num_Procedures': 'Procedure Count',
        **{f'Procedure_{code}': f'Procedure {code}' for code in ['P1','P2','P3','P4','P5','P6','P7']}
    }
    influence_df['Feature_Name'] = influence_df['Feature'].map(feature_map)

    fig = go.Figure(go.Bar(
        x=influence_df['SHAP_Value'],
        y=influence_df['Feature_Name'],
        orientation='h',
        marker_color=['red' if x > np.mean(influence_df['SHAP_Value']) else 'blue' 
                     for x in influence_df['SHAP_Value']],
        text=influence_df['Feature_Name'],
        textposition='auto'
    ))

    fig.update_layout(
        title=f"DRG {drg_code} Risk Drivers Analysis",
        xaxis_title="Feature Impact (Mean Absolute SHAP)",
        yaxis_title="Features",
        template='plotly_white',
        height=700,
        margin=dict(l=150, r=50, b=100, t=100)
    )

    return (nlg_summary, fig, queries)

# Example execution with error handling
def analyze_drg(drg_code):
    result = explain_overpayment_for_drg(drg_code)
    
    if isinstance(result, tuple):
        summary, plot, queries = result
        
        print("\n" + "="*80)
        print(summary)
        
        print("\n" + "="*80)
        print("🛠️ Recommended Audit Actions:")
        for i, q in enumerate(queries, 1):
            print(f"{i}. {q}")
            
        plot.show()
    else:
        print(f"⚠️ {result}")

# Run analysis for multiple DRGs
for drg in ['039', '470', '123', '456']:
    analyze_drg(drg)

