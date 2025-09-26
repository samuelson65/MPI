import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import shap

# =============================
# 1. Train sample model
# =============================

# Example dataset (replace with Medicare claims data)
df = pd.DataFrame({
    "diagnosis_code": [1, 1, 2, 2, 3, 3],
    "procedure_code": [5, 4, 5, 6, 4, 6],
    "length_of_stay": [5, 7, 12, 3, 15, 6],
    "total_charges": [5000, 7000, 20000, 4000, 25000, 6000],
    "flagged": [0, 0, 1, 0, 1, 0]  # 1 = suspicious claim
})

X = df.drop(columns=["flagged"])
y = df["flagged"]

# Train simple model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)[1]  # focus on "flagged=1" class

# =============================
# 2. Explanations
# =============================

def shap_nlg_explanation(claim_id, feature_names, shap_values, X):
    """Generate layman explanation using SHAP."""
    values = shap_values[claim_id]
    features = X.iloc[claim_id]

    # Top 2 important features
    top_idx = np.argsort(np.abs(values))[::-1][:2]
    explanations = []

    for idx in top_idx:
        feat = feature_names[idx]
        val = features.iloc[idx]

        if values[idx] > 0:
            explanations.append(f"{feat.replace('_',' ')} is unusually high ({val}), increasing risk.")
        else:
            explanations.append(f"{feat.replace('_',' ')} looks typical ({val}), lowering concern.")

    return " ".join(explanations)

def counterfactual_explanation(claim_id, feature_names, shap_values, X):
    """Suggest how the claim could look more normal."""
    values = shap_values[claim_id]
    features = X.iloc[claim_id]
    top_idx = np.argsort(np.abs(values))[::-1][:1]  # strongest feature

    feat = feature_names[top_idx[0]]
    val = features.iloc[top_idx[0]]

    if values[top_idx[0]] > 0:
        return f"If the {feat.replace('_',' ')} were closer to typical values instead of {val}, this claim would appear more routine."
    else:
        return f"Even if the {feat.replace('_',' ')} were different, it wouldn’t change much — the claim already looks fairly normal."

def peer_explanation(claim_id, X, df, top_n=3):
    """Compare claim to nearest neighbors."""
    sims = cosine_similarity(X.iloc[claim_id].values.reshape(1, -1), X)[0]
    neighbor_ids = sims.argsort()[::-1][1:top_n+1]
    neighbors = df.iloc[neighbor_ids]

    claim = df.iloc[claim_id]
    avg_los = neighbors["length_of_stay"].mean()
    avg_charge = neighbors["total_charges"].mean()

    return (
        f"Similar claims usually had a hospital stay of {avg_los:.1f} days "
        f"and charges around ${avg_charge:,.0f}. "
        f"This claim had {claim['length_of_stay']} days and charges of ${claim['total_charges']:,.0f}, "
        f"which makes it stand out from its peers."
    )

# =============================
# 3. Run explanations for one claim
# =============================
claim_id = 2  # example claim index

print("🔎 SHAP Narrative:")
print(shap_nlg_explanation(claim_id, X.columns, shap_values, X))
print("\n💡 Counterfactual:")
print(counterfactual_explanation(claim_id, X.columns, shap_values, X))
print("\n📊 Peer Comparison:")
print(peer_explanation(claim_id, X, df))
