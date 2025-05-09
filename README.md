import pandas as pd
import shap
import numpy as np
from catboost import CatBoostClassifier

# Load model and data
model = CatBoostClassifier()
model.load_model("your_model.cbm")  # Path to your CatBoost model

X = pd.read_csv("your_data.csv")  # Input features

# Get SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Function to extract top SHAP contributors
def explain_shap_values(shap_row, feature_names, instance, top_n=3):
    shap_contribs = list(zip(feature_names, shap_row, instance))
    shap_contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    top_pos = [f for f in shap_contribs if f[1] > 0][:top_n]
    top_neg = [f for f in shap_contribs if f[1] < 0][:top_n]
    return top_pos, top_neg

# Smarter NLG generator
def generate_nlg_summary(top_pos, top_neg):
    phrases = []

    if top_pos:
        reasons = []
        for name, _, val in top_pos:
            reasons.append(f"{name} being elevated ({val})")
        phrases.append("The prediction indicates a higher risk of preventable readmission, likely due to " +
                       ", ".join(reasons) + ".")

    if top_neg:
        protectors = []
        for name, _, val in top_neg:
            protectors.append(f"{name} being relatively low ({val})")
        phrases.append("However, there are protective factors such as " +
                       ", ".join(protectors) + " that lower the overall risk.")

    if not phrases:
        return "No strong predictors were identified for this case."

    return " ".join(phrases)

# Apply to each row
summaries = []
for i in range(len(X)):
    top_pos, top_neg = explain_shap_values(shap_values[i], X.columns, X.iloc[i])
    summary = generate_nlg_summary(top_pos, top_neg)
    summaries.append(summary)

# Add to DataFrame
X['readmission_nlg_summary'] = summaries

# Optional: Save output
X.to_csv("readmission_with_smart_summary.csv", index=False)
