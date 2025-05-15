import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from rulefit import RuleFit
import shap

# 1. Create sample DRG dataset
data = {
    'age': [65, 72, 58, 80, 45, 50, 77, 68, 59, 83],
    'sex': [1, 0, 1, 0, 1, 0, 1, 1, 0, 0],  # 1=Male, 0=Female
    'primary_diagnosis_code': [101, 102, 103, 101, 104, 105, 102, 103, 104, 101],
    'num_secondary_diagnoses': [2, 5, 1, 3, 0, 4, 3, 2, 1, 6],
    'num_procedures': [1, 3, 0, 2, 1, 0, 4, 2, 1, 3],
    'length_of_stay': [5, 14, 3, 10, 2, 4, 12, 7, 3, 15],
    'severity_of_illness': [3, 4, 2, 4, 1, 2, 4, 3, 2, 5],  # scale 1-5
    'label': [1, 1, 0, 1, 0, 0, 1, 0, 0, 1]  # 1 = High resource DRG, 0 = Low resource DRG
}
df = pd.DataFrame(data)

# Features and target
X = df.drop(columns=['label'])
y = df['label']

# 2. Train Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 3. Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)[1]  # SHAP values for class 1

# 4. Cluster SHAP values to find subgroups
cluster_model = AgglomerativeClustering(n_clusters=2)
clusters = cluster_model.fit_predict(shap_values)
df['shap_cluster'] = clusters

# Add cluster as a feature for RuleFit
X['shap_cluster'] = clusters

# 5. Fit RuleFit model to extract interpretable rules
rf = RuleFit(tree_size=4, sample_fract='default', max_rules=10, memory_par=0.01, random_state=42)
rf.fit(X.values, y.values, feature_names=X.columns)

# 6. Extract and print important rules (conditional queries)
rules = rf.get_rules()
rules = rules[(rules.coef != 0) & (rules.type == 'rule')].sort_values(by='support', ascending=False)

print("\n=== Extracted Conditional Queries (Rules) ===")
for _, row in rules.iterrows():
    print(f"Rule: {row['rule']}")
    print(f"  Coefficient: {row['coef']:.3f}")
    print(f"  Support: {row['support']:.3f}\n")
