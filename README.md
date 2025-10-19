import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

def classify_drg_clusters(df,
                          random_state=42,
                          min_clusters=2,
                          max_clusters=8,
                          return_centroids=True,
                          include_action=True):
    """
    Cluster DRG codes (drg_code like 'drg_100') and produce aligned, readable cluster labels.

    Returns:
        drg_cluster_df: DataFrame with columns ['drg_code', 'cluster', 'cluster_label', 'action' (optional)]
        cluster_summary_df: DataFrame with cluster means and cluster_label (if return_centroids=True)
    """

    # --- 1. Validate & normalize input ---
    required_cols = ['drg_code', 'median_probability', 'hitrate', 'avg_overpayment', 'volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # ensure drg_code is string and normalized
    df['drg_code'] = df['drg_code'].astype(str)

    # numeric conversions
    num_cols = ['median_probability', 'hitrate', 'avg_overpayment', 'volume']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    # drop rows that lost numeric values
    df = df.dropna(subset=num_cols).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows after cleaning numeric columns.")

    # adjust max_clusters for tiny datasets
    max_clusters = min(max_clusters, max(2, len(df) // 2))

    # --- 2. Scale features for clustering ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[num_cols])

    # --- 3. Find optimal k using silhouette score (safe) ---
    silhouette_scores = {}
    for k in range(min_clusters, max_clusters + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                labels = km.fit_predict(X_scaled)
            # silhouette requires at least 2 labels; skip if k=1
            if len(set(labels)) > 1:
                silhouette_scores[k] = silhouette_score(X_scaled, labels)
        except Exception:
            continue

    if silhouette_scores:
        optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    else:
        optimal_k = min(3, max(2, len(df)))  # fallback

    # --- 4. Fit final KMeans and assign clusters ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_km = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
        df['cluster'] = final_km.fit_predict(X_scaled)

    # --- 5. Compute cluster summary (in ORIGINAL data space) to guarantee alignment ---
    cluster_summary_df = df.groupby('cluster')[num_cols].mean().reset_index()

    # --- 6. Create descriptive labels based on medians of the dataset ---
    medians = df[num_cols].median()

    def make_label(row):
        parts = []
        parts.append("High Hitrate" if row['hitrate'] > medians['hitrate'] else "Low Hitrate")
        parts.append("High Overpayment" if row['avg_overpayment'] > medians['avg_overpayment'] else "Low Overpayment")
        parts.append("High Volume" if row['volume'] > medians['volume'] else "Low Volume")
        parts.append("High Probability" if row['median_probability'] > medians['median_probability'] else "Low Probability")
        return " & ".join(parts)

    cluster_summary_df['cluster_label'] = cluster_summary_df.apply(make_label, axis=1)

    # --- 7. Merge labels back onto DRG-level df (guaranteed aligned) ---
    df = df.merge(cluster_summary_df[['cluster', 'cluster_label']], on='cluster', how='left')

    # --- 8. Optional actionable flag ---
    if include_action:
        # define simple rule: strong include if cluster has High Hitrate & High Overpayment & High Volume
        def action_from_label(lbl):
            if all(x in lbl for x in ["High Hitrate", "High Overpayment", "High Volume"]):
                return "Include More"
            if all(x in lbl for x in ["Low Hitrate", "Low Overpayment", "Low Volume"]):
                return "Exclude"
            return "Review"

        df['action'] = df['cluster_label'].apply(action_from_label)

    # --- 9. Final tidy outputs ---
    out_cols = ['drg_code', 'cluster', 'cluster_label']
    if include_action:
        out_cols.append('action')
    drg_cluster_df = df[out_cols].copy().sort_values(['cluster', 'drg_code']).reset_index(drop=True)

    # reorder cluster_summary cols nicely
    cluster_summary_df = cluster_summary_df[['cluster', 'cluster_label'] + num_cols]

    if return_centroids:
        return drg_cluster_df, cluster_summary_df
    else:
        return drg_cluster_df


# ---------------- Example usage ----------------
if __name__ == "__main__":
    sample = pd.DataFrame({
        'drg_code': ['drg_100', 'drg_101', 'drg_102', 'drg_103', 'drg_104', 'drg_105', 'drg_106'],
        'median_probability': [0.82, 0.60, 0.45, 0.91, 0.38, 0.75, 0.52],
        'hitrate':           [0.70, 0.55, 0.25, 0.88, 0.20, 0.68, 0.33],
        'avg_overpayment':   [5800, 3200, 1100, 7200, 800, 4500, 2000],
        'volume':            [45, 120, 300, 25, 400, 95, 60]
    })

    drg_map, cluster_summary = classify_drg_clusters(sample, return_centroids=True)
    print("\n=== DRG to Cluster Mapping ===")
    print(drg_map)
    print("\n=== Cluster Summary (centroids in original space) ===")
    print(cluster_summary)
