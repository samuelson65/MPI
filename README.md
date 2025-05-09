import pandas as pd

# Inputs: shap_df = SHAP values DataFrame, X = feature values DataFrame (same index & columns)
# Example:
# shap_df = pd.read_csv("shap_values.csv")
# X = pd.read_csv("original_features.csv")

def explain_shap_row(shap_row, feature_row, top_n=3):
    contribs = list(zip(shap_row.index, shap_row.values, feature_row.values))
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    top_pos = [f for f in contribs if f[1] > 0][:top_n]
    top_neg = [f for f in contribs if f[1] < 0][:top_n]
    return top_pos, top_neg

def generate_nlg_summary(top_pos, top_neg):
    summary_parts = []

    if top_pos:
        reasons = [f"{name} being elevated ({val})" for name, _, val in top_pos]
        summary_parts.append("This patient's predicted risk of readmission is high, primarily due to " +
                             ", ".join(reasons) + ".")

    if top_neg:
        protectors = [f"{name} being relatively low ({val})" for name, _, val in top_neg]
        summary_parts.append("However, protective factors such as " +
                             ", ".join(protectors) + " may mitigate the risk.")

    if not summary_parts:
        return "No dominant factors were identified for this prediction."

    return " ".join(summary_parts)

# Main loop
summaries = []
for i in range(len(shap_df)):
    top_pos, top_neg = explain_shap_row(shap_df.iloc[i], X.iloc[i])
    summary = generate_nlg_summary(top_pos, top_neg)
    summaries.append(summary)

# Add to original DataFrame
X = X.copy()
X['readmission_nlg_summary'] = summaries

# Optional: Save
# X.to_csv("with_shap_summaries.csv", index=False)
