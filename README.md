import pandas as pd
import numpy as np

# ========== SHAP Explanation Helpers ==========
def explain_shap_row(shap_row, feature_row, top_n=3):
    contribs = list(zip(shap_row.index, shap_row.values, feature_row.values))
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    top_pos = [f for f in contribs if f[1] > 0][:top_n]
    top_neg = [f for f in contribs if f[1] < 0][:top_n]
    return top_pos, top_neg

def format_feature_phrases(features, direction="elevated", plain=False):
    phrases = []
    for name, _, val in features:
        name_clean = name.replace('_', ' ').capitalize()
        if plain:
            phr = f"{name_clean} is {direction} (value: {val})"
        else:
            phr = f"{name.replace('_', ' ')} being {direction} ({val})"
        phrases.append(phr)
    return phrases

# ========== Summary Generators ==========
def generate_clinical_summary(top_pos, top_neg, pred_prob, threshold=0.5):
    summary = []
    if pred_prob >= threshold + 0.1:
        summary.append(f"High readmission risk (p={pred_prob:.2f}).")
    elif pred_prob <= threshold - 0.1:
        summary.append(f"Low readmission risk (p={pred_prob:.2f}).")
    else:
        summary.append(f"Moderate readmission risk (p={pred_prob:.2f}).")
    if top_pos:
        summary.append("Top contributors: " + ", ".join(format_feature_phrases(top_pos, "elevated")) + ".")
    if top_neg:
        summary.append("Protective factors: " + ", ".join(format_feature_phrases(top_neg, "low")) + ".")
    return " ".join(summary)

def generate_layman_summary(top_pos, top_neg, pred_prob, threshold=0.5):
    summary = []
    if pred_prob >= threshold + 0.1:
        summary.append(f"Your risk of being readmitted to the hospital is high (estimated chance: {int(pred_prob * 100)}%).")
    elif pred_prob <= threshold - 0.1:
        summary.append(f"Your chance of being readmitted is low (about {int(pred_prob * 100)}%).")
    else:
        summary.append(f"You have a moderate chance of readmission (around {int(pred_prob * 100)}%).")
    if top_pos:
        summary.append("This is mainly due to " + ", ".join(format_feature_phrases(top_pos, "high", plain=True)) + ".")
    if top_neg:
        summary.append("On the positive side, factors like " + ", ".join(format_feature_phrases(top_neg, "low", plain=True)) + " may reduce your risk.")
    return " ".join(summary)

# ========== Main Processor ==========
def generate_shap_summaries_dual(shap_df, X, pred_proba, threshold=0.5, top_n=3):
    clinical_summaries = []
    patient_summaries = []

    for i in range(len(X)):
        shap_row = shap_df.iloc[i]
        feature_row = X.iloc[i]
        prob = pred_proba[i]
        top_pos, top_neg = explain_shap_row(shap_row, feature_row, top_n=top_n)

        clinical = generate_clinical_summary(top_pos, top_neg, prob, threshold)
        patient = generate_layman_summary(top_pos, top_neg, prob, threshold)

        clinical_summaries.append(clinical)
        patient_summaries.append(patient)

    X = X.copy()
    X['clinical_summary'] = clinical_summaries
    X['patient_summary'] = patient_summaries
    return X

# ========== Example Usage ==========
# shap_df = pd.read_csv("shap_values.csv")     # SHAP values: one row per patient
# X = pd.read_csv("original_features.csv")     # Feature values
# pred_proba = model.predict_proba(X)[:, 1]    # Predicted readmission probabilities

# result_df = generate_shap_summaries_dual(shap_df, X, pred_proba)
# result_df.to_csv("readmission_with_summaries.csv", index=False)
