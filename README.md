import pandas as pd
import shap
import numpy as np
from catboost import CatBoostClassifier

# Step 1: Load model and data
# Replace with your actual model and dataset
model = CatBoostClassifier()
model.load_model("your_model.cbm")  # path to your model file

X = pd.read_csv("your_data.csv")  # your input data

# Step 2: Get SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Step 3: Function to extract top positive and negative features
def explain_shap_values(shap_row, feature_names, instance, top_n=3):
    shap_contribs = list(zip(feature_names, shap_row, instance))
    shap_contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    top_pos = [f for f in shap_contribs if f[1] > 0][:top_n]
    top_neg = [f for f in shap_contribs if f[1] < 0][:top_n]
    return top_pos, top_neg

# Step 4: Generate NLG summary
def generate_nlg_summary(top_pos, top_neg):
    summary = ""
    if top_pos:
        summary += "The risk of preventable readmission is high mainly due to "
        summary += ", ".join([f"high {name} ({val})" for name, _, val in top_pos]) + ". "
    if top_neg:
        summary += "However, the risk is reduced because of "
        summary += ", ".join([f"low {name} ({val})" for name, _, val in top_neg]) + "."
    return summary.strip()

# Step 5: Apply across all rows
summaries = []
for i in range(len(X)):
    top_pos, top_neg = explain_shap_values(shap_values[i], X.columns, X.iloc[i])
    summary = generate_nlg_summary(top_pos, top_neg)
    summaries.append(summary)

# Step 6: Add to DataFrame
X['readmission_nlg_summary'] = summaries

# Optional: Save to file
X.to_csv("readmission_with_summary.csv", index=False)
