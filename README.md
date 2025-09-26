import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def explain_claim(shap_row, feature_row, pred_prob, X_all, top_n_neighbors=3):
    """
    Generate explanations for a single claim.

    Inputs:
    - shap_row: array-like of SHAP values for this claim (1D, len = features)
    - feature_row: pandas Series or dict of feature values for this claim
    - pred_prob: predicted probability of being flagged (float, 0–1)
    - X_all: DataFrame of all claims' features (for peer comparison)
    - top_n_neighbors: number of nearest neighbors for similarity

    Returns:
    - dict with 'narrative', 'counterfactual', 'peer' explanations
    """
    
    # Ensure feature_row is a Series
    if isinstance(feature_row, dict):
        feature_row = pd.Series(feature_row)
    
    feature_names = feature_row.index.tolist()
    shap_row = np.array(shap_row)

    explanations = {}

    # ======================
    # 1. SHAP Narrative
    # ======================
    top_idx = np.argsort(np.abs(shap_row))[::-1][:2]  # top 2 drivers
    parts = []
    
    for idx in top_idx:
        feat = feature_names[idx]
        val = feature_row.iloc[idx]
        if shap_row[idx] > 0:
            parts.append(f"{feat.replace('_',' ')} is unusually high ({val}), increasing risk.")
        else:
            parts.append(f"{feat.replace('_',' ')} looks typical ({val}), lowering concern.")
    
    narrative = " ".join(parts)
    explanations["narrative"] = (
        f"The model predicts this claim has a {pred_prob:.0%} chance of being flagged. {narrative}"
    )

    # ======================
    # 2. Counterfactual
    # ======================
    top_idx = np.argsort(np.abs(shap_row))[::-1][:1]  # strongest feature
    feat = feature_names[top_idx[0]]
    val = feature_row.iloc[top_idx[0]]
    
    if shap_row[top_idx[0]] > 0:
        counterfactual = (
            f"If the {feat.replace('_',' ')} were closer to typical values instead of {val}, "
            f"this claim would appear more routine."
        )
    else:
        counterfactual = (
            f"Even if the {feat.replace('_',' ')} were different, it wouldn’t change much — "
            f"the claim already looks fairly normal."
        )
    
    explanations["counterfactual"] = counterfactual

    # ======================
    # 3. Peer Explanation
    # ======================
    claim_idx = feature_row.name if feature_row.name in X_all.index else None
    
    if claim_idx is not None:
        sims = cosine_similarity(feature_row.values.reshape(1, -1), X_all.values)[0]
        neighbor_ids = sims.argsort()[::-1][1:top_n_neighbors+1]
        neighbors = X_all.iloc[neighbor_ids]
        
        # Pick two numeric columns for comparison
        numeric_cols = X_all.select_dtypes(include=np.number).columns
        peer_parts = []
        for col in numeric_cols[:2]:  # first 2 numeric cols
            avg_val = neighbors[col].mean()
            peer_parts.append(
                f"Similar claims usually had {col.replace('_',' ')} around {avg_val:.1f}, "
                f"but this claim had {feature_row[col]}."
            )
        
        peer_expl = " ".join(peer_parts)
    else:
        peer_expl = "Peer comparison not available (index mismatch)."
    
    explanations["peer"] = peer_expl

    return explanations


# ======================
# Example usage
# ======================
if __name__ == "__main__":
    # Example SHAP values for one claim (4 features)
    shap_row = np.array([0.2, -0.1, 0.5, 0.3])
    
    # Example claim feature values
    feature_row = pd.Series({
        "length_of_stay": 12,
        "total_charges": 20000,
        "diagnosis_code": 2,
        "procedure_code": 5
    }, name=2)
    
    # Example predicted probability
    pred_prob = 0.87
    
    # Example dataset for similarity lookup
    X_all = pd.DataFrame({
        "length_of_stay": [5, 7, 12, 3, 15, 6],
        "total_charges": [5000, 7000, 20000, 4000, 25000, 6000],
        "diagnosis_code": [1, 1, 2, 2, 3, 3],
        "procedure_code": [5, 4, 5, 6, 4, 6]
    })
    
    # Run explanation
    explanations = explain_claim(shap_row, feature_row, pred_prob, X_all)
    
    print("🔎 Narrative:", explanations["narrative"])
    print("💡 Counterfactual:", explanations["counterfactual"])
    print("📊 Peer:", explanations["peer"])
