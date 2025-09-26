import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------- SHAP Explanation Helpers ----------

def explain_shap_row(shap_row, feature_row, top_n=3):
    """Extract top positive and negative contributors"""
    contribs = list(zip(shap_row.index, shap_row.values, feature_row.values))
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    top_pos = [(f, v, val) for f, v, val in contribs if v > 0][:top_n]
    top_neg = [(f, v, val) for f, v, val in contribs if v < 0][:top_n]
    return top_pos, top_neg


def generate_layman_summary(top_pos, top_neg, pred_prob, threshold=0.5):
    """Generate natural language explanation for layman"""
    summary = []
    if pred_prob > threshold + 0.1:
        summary.append(
            f"Your claim has a high chance of being flagged (about {int(pred_prob*100)}%)."
        )
    elif pred_prob < threshold - 0.1:
        summary.append(
            f"Your claim has a low chance of being flagged (around {int(pred_prob*100)}%)."
        )
    else:
        summary.append(
            f"Your claim has a moderate chance of being flagged (around {int(pred_prob*100)}%)."
        )

    if top_pos:
        pos_feats = ", ".join([f"{f} being high" for f, _, val in top_pos])
        summary.append(f"This is mainly due to {pos_feats}.")
    if top_neg:
        neg_feats = ", ".join([f"{f} being low" for f, _, val in top_neg])
        summary.append(f"On the positive side, factors like {neg_feats} reduce your risk.")

    return " ".join(summary)


def generate_counterfactual_summary(top_pos, top_neg, pred_prob, threshold=0.5):
    """Suggest counterfactual: what needs to change"""
    summary = []
    if pred_prob > threshold + 0.1:
        summary.append("To reduce the chance of being flagged, changes could include:")
        if top_pos:
            for f, _, val in top_pos:
                summary.append(f"- Lowering {f} (currently {val})")
    elif pred_prob < threshold - 0.1:
        summary.append("Your claim is safe, but risk could rise if:")
        if top_neg:
            for f, _, val in top_neg:
                summary.append(f"- {f} were higher (currently {val})")
    else:
        summary.append("Small changes could shift this claim either way.")
        if top_pos:
            summary.append(
                f"For example, lowering {top_pos[0][0]} could reduce risk further."
            )
        if top_neg:
            summary.append(
                f"Whereas increasing {top_neg[0][0]} could make the claim more risky."
            )

    return " ".join(summary)


def generate_similarity_explanations(input_df, shap_df, top_k=1):
    """
    Create similarity-based explanations using:
    - Shared features
    - Diverging SHAP contributions
    """
    numeric_cols = input_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["probabilityscore"]]

    if not numeric_cols:
        return ["No numeric features available for similarity analysis."] * len(input_df)

    # similarity matrix
    X_num = input_df[numeric_cols].fillna(0).values
    sim_matrix = cosine_similarity(X_num)

    explanations = []
    for i in range(len(input_df)):
        sims = list(enumerate(sim_matrix[i]))
        sims = sorted(sims, key=lambda x: -x[1])
        sims = [j for j in sims if j[0] != i]

        top_matches = sims[:top_k]
        claim = input_df.iloc[i]
        claim_pk = claim["pk"]
        claim_prob = claim["probabilityscore"]
        claim_shap = shap_df.iloc[i]

        claim_exp = []
        for idx, score in top_matches:
            peer = input_df.iloc[idx]
            peer_pk = peer["pk"]
            peer_prob = peer["probabilityscore"]
            peer_shap = shap_df.iloc[idx]

            # shared features
            similar_features = []
            for col in input_df.columns:
                if col in ["pk", "probabilityscore"]:
                    continue
                if str(claim[col]) == str(peer[col]):
                    similar_features.append(f"{col} ({claim[col]})")

            # shap-driven differences
            shap_diff = (claim_shap - peer_shap).abs().sort_values(ascending=False)
            top_diff = shap_diff.head(2).index.tolist()

            diff_texts = []
            for col in top_diff:
                diff_texts.append(
                    f"{col} influenced claim {claim_pk} more ({claim_shap[col]:.2f}) "
                    f"than claim {peer_pk} ({peer_shap[col]:.2f})"
                )

            # build explanation
            sim_text = (
                f"Claim {claim_pk} is most similar to claim {peer_pk}, "
                f"sharing factors like {', '.join(similar_features[:3])}."
            )

            if peer_prob != claim_prob and diff_texts:
                diff_text = (
                    f" Their risk scores differ ({claim_prob:.2f} vs {peer_prob:.2f}) "
                    f"mainly because {', '.join(diff_texts)}."
                )
            else:
                diff_text = (
                    f" Their risk scores are also quite similar "
                    f"({claim_prob:.2f} vs {peer_prob:.2f})."
                )

            claim_exp.append(sim_text + " " + diff_text)

        explanations.append(" ".join(claim_exp))

    return explanations


# ---------- Main Processor ----------

def generate_explanations(input_df, shap_df, threshold=0.5, top_n=3):
    layman_summaries = []
    counterfactuals = []

    for i in range(len(input_df)):
        shap_row = shap_df.iloc[i]
        feature_row = input_df.drop(columns=["pk", "probabilityscore"]).iloc[i]
        prob = input_df.iloc[i]["probabilityscore"]

        top_pos, top_neg = explain_shap_row(shap_row, feature_row, top_n=top_n)

        layman = generate_layman_summary(top_pos, top_neg, prob, threshold)
        counterfactual = generate_counterfactual_summary(top_pos, top_neg, prob, threshold)

        layman_summaries.append(layman)
        counterfactuals.append(counterfactual)

    # similarity explanations
    similarity_exps = generate_similarity_explanations(input_df, shap_df, top_k=1)

    # attach results
    result_df = input_df.copy()
    result_df["layman_explanation"] = layman_summaries
    result_df["counterfactual_explanation"] = counterfactuals
    result_df["similarity_explanation"] = similarity_exps

    return result_df


# ---------- Example Usage ----------
if __name__ == "__main__":
    # fake example
    input_df = pd.DataFrame({
        "pk": [101, 102, 103],
        "Age": [65, 70, 65],
        "Gender": ["M", "F", "M"],
        "Length_of_stay": [12, 5, 10],
        "Comorbidity_count": [2, 0, 3],
        "probabilityscore": [0.75, 0.35, 0.60]
    })

    shap_df = pd.DataFrame([
        {"Age": 0.02, "Gender": 0.01, "Length_of_stay": 0.35, "Comorbidity_count": 0.20},
        {"Age": -0.01, "Gender": 0.00, "Length_of_stay": 0.05, "Comorbidity_count": -0.05},
        {"Age": 0.01, "Gender": 0.00, "Length_of_stay": 0.25, "Comorbidity_count": 0.10},
    ])

    result = generate_explanations(input_df, shap_df)
    print(result[["pk", "layman_explanation", "counterfactual_explanation", "similarity_explanation"]])
