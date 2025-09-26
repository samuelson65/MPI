import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# SHAP Explanation Helpers
# =========================
def explain_shap_row(shap_row, feature_row, top_n=3):
    contribs = list(zip(shap_row.index, shap_row.values, feature_row.values))
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    top_pos = [(f, v) for f, s, v in contribs if s > 0][:top_n]
    top_neg = [(f, v) for f, s, v in contribs if s < 0][:top_n]
    return top_pos, top_neg

def format_feature_phrases(features, direction="elevated", plain=False):
    phrases = []
    for name, val in features:
        name_clean = name.replace("_", " ").capitalize()
        if plain:
            phr = f"{name_clean} is {direction} (value: {val})"
        else:
            phr = f"{name_clean} being {direction} ({val})"
        phrases.append(phr)
    return phrases

# =========================
# Clinical Explanation
# =========================
def generate_clinical_summary(top_pos, top_neg, pred_prob, threshold=0.5):
    summary = []
    if pred_prob > threshold + 0.1:
        summary.append(f"High risk (p={pred_prob:.2f}).")
    elif pred_prob <= threshold - 0.1:
        summary.append(f"Low risk (p={pred_prob:.2f}).")
    else:
        summary.append(f"Moderate risk (p={pred_prob:.2f}).")

    if top_pos:
        summary.append("Top contributors: " + ", ".join(format_feature_phrases(top_pos, "elevated")) + ".")
    if top_neg:
        summary.append("Protective factors: " + ", ".join(format_feature_phrases(top_neg, "low")) + ".")
    return " ".join(summary)

# =========================
# Layman Explanation
# =========================
def generate_layman_summary(top_pos, top_neg, pred_prob, threshold=0.5):
    summary = []
    if pred_prob > threshold + 0.1:
        summary.append(f"Your risk is high (about {int(pred_prob*100)}%).")
    elif pred_prob < threshold - 0.1:
        summary.append(f"Your chance of risk is low (around {int(pred_prob*100)}%).")
    else:
        summary.append(f"You have a moderate chance (around {int(pred_prob*100)}%).")

    if top_pos:
        summary.append("This is mainly due to " + ", ".join(format_feature_phrases(top_pos, "high", plain=True)) + ".")
    if top_neg:
        summary.append("On the positive side, factors like " + ", ".join(format_feature_phrases(top_neg, "low", plain=True)) + " help reduce the risk.")
    return " ".join(summary)

# =========================
# Counterfactual Explanation
# =========================
def generate_counterfactual(top_pos, top_neg, pred_prob, threshold=0.5):
    summary = []
    if pred_prob > threshold:
        if top_neg:
            summary.append("If protective factors like " + ", ".join([f for f, _ in top_neg]) + " were stronger, the claim could have been classified as lower risk.")
        else:
            summary.append("If one or more top contributing factors were reduced, this claim might shift to lower risk.")
    else:
        if top_pos:
            summary.append("If factors such as " + ", ".join([f for f, _ in top_pos]) + " were higher, the claim might move towards higher risk.")
        else:
            summary.append("With stronger negative signals, this claim could shift upward in risk.")
    return " ".join(summary)

# =========================
# Similarity-based Explanation
# =========================
def generate_similarity_explanations(input_df, top_k=1):
    # Select only numeric columns for similarity
    numeric_cols = input_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["probabilityscore"]]  # exclude prob itself

    if not numeric_cols:  # if no numeric features, fallback
        return ["No numeric features available for similarity analysis."] * len(input_df)

    X_num = input_df[numeric_cols].fillna(0).values
    sim_matrix = cosine_similarity(X_num)

    explanations = []
    for i in range(len(input_df)):
        sims = list(enumerate(sim_matrix[i]))
        sims = sorted(sims, key=lambda x: -x[1])
        sims = [j for j in sims if j[0] != i]

        top_matches = sims[:top_k]
        phrases = []
        for idx, score in top_matches:
            pk_match = input_df.iloc[idx]["pk"]
            prob_match = input_df.iloc[idx]["probabilityscore"]
            phrases.append(f"claim {pk_match} (prob {prob_match:.2f})")

        explanations.append(f"This claim is most similar to {', '.join(phrases)}.")
    return explanations

# =========================
# Main Processor
# =========================
def generate_explanations(shap_df, input_df, threshold=0.5, top_n=3, top_k_sim=1):
    clinical_summaries = []
    patient_summaries = []
    counterfactuals = []

    # Features (exclude pk + probabilityscore)
    feature_cols = [c for c in input_df.columns if c not in ["pk", "probabilityscore"]]

    for i in range(len(input_df)):
        shap_row = shap_df.iloc[i]
        feature_row = input_df[feature_cols].iloc[i]
        prob = input_df.iloc[i]["probabilityscore"]

        top_pos, top_neg = explain_shap_row(shap_row, feature_row, top_n=top_n)

        clinical = generate_clinical_summary(top_pos, top_neg, prob, threshold)
        patient = generate_layman_summary(top_pos, top_neg, prob, threshold)
        counterf = generate_counterfactual(top_pos, top_neg, prob, threshold)

        clinical_summaries.append(clinical)
        patient_summaries.append(patient)
        counterfactuals.append(counterf)

    similarity_summaries = generate_similarity_explanations(input_df, top_k=top_k_sim)

    # Attach results
    result_df = input_df.copy()
    result_df["clinical_summary"] = clinical_summaries
    result_df["patient_summary"] = patient_summaries
    result_df["counterfactual_summary"] = counterfactuals
    result_df["similarity_summary"] = similarity_summaries
    return result_df

# =========================
# Example Usage
# =========================
if __name__ == "__main__":
    # Example shap values
    shap_df = pd.DataFrame(
        [[0.2, -0.1, 0.05], [-0.3, 0.2, -0.05]],
        columns=["age", "comorbidity", "length_of_stay"]
    )

    # Input data with pk, probabilityscore, numeric + categorical features
    input_df = pd.DataFrame(
        {
            "pk": [101, 102],
            "probabilityscore": [0.75, 0.35],
            "age": [65, 45],
            "comorbidity": [2, 0],
            "length_of_stay": [5, 2],
            "diagnosis": ["Diabetes", "Hypertension"],   # string feature
            "gender": ["M", "F"]                        # string feature
        }
    )

    result_df = generate_explanations(shap_df, input_df, top_k_sim=2)
    print(result_df[["pk", "probabilityscore", "clinical_summary", "patient_summary", "counterfactual_summary", "similarity_summary"]])
