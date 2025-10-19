import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

def _centroid_level_from_percentiles(centroid_vals, n_clusters):
    """
    Given an array-like of centroid values for a metric, return levels per cluster:
    - If n_clusters > 3: use percentile cutoffs (<=33% -> Low, 33-66 -> Medium, >66 -> High)
    - If n_clusters <= 3: use strict ranking (bottom -> Low, middle -> Medium, top -> High)
    Returns a list of levels aligned to centroid index order.
    """
    arr = np.asarray(centroid_vals)
    levels = []
    if n_clusters > 3:
        p33 = np.percentile(arr, 33)
        p66 = np.percentile(arr, 66)
        for v in arr:
            if v <= p33:
                levels.append("Low")
            elif v <= p66:
                levels.append("Medium")
            else:
                levels.append("High")
    else:
        # rank-based fallback
        ranks = arr.argsort().argsort()  # 0..n-1 in increasing order
        for r in ranks:
            if r == 0:
                levels.append("Low")
            elif r == n_clusters - 1:
                levels.append("High")
            else:
                levels.append("Medium")
    return levels

def classify_drg_clusters(df,
                          random_state=42,
                          min_clusters=2,
                          max_clusters=8,
                          return_centroids=True,
                          include_action=True):
    """
    Cluster DRG codes (drg_code like 'drg_100') using hitrate, avg_overpayment, and volume.
    Produces aligned cluster labels using centroid-relative levels (Low/Medium/High).

    Parameters:
        df (pd.DataFrame): Must contain ['drg_code', 'hitrate', 'avg_overpayment', 'volume']
        random_state (int): Random seed for reproducibility.
        min_clusters (int): Minimum clusters to evaluate.
        max_clusters (int): Maximum clusters to evaluate.
        return_centroids (bool): If True, return (drg_cluster_df, cluster_summary_df)
        include_action (bool): If True, include 'action' column with Include More / Exclude / Review.

    Returns:
        drg_cluster_df (pd.DataFrame) or (drg_cluster_df, cluster_summary_df)
    """

    # --- Validate input columns ---
    required = ['drg_code', 'hitrate', 'avg_overpayment', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df['drg_code'] = df['drg_code'].astype(str)

    # --- Numeric conversion & cleaning ---
    features = ['hitrate', 'avg_overpayment', 'volume']
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=features).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid numeric rows after cleaning.")

    # --- Adjust cluster search range for tiny datasets ---
    max_clusters = min(max_clusters, max(2, len(df) // 2))

    # --- Scale features for clustering ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # --- Find optimal k with silhouette (safe loop) ---
    silhouette_scores = {}
    for k in range(min_clusters, max_clusters + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
                labels = km.fit_predict(X_scaled)
            if len(set(labels)) > 1:
                silhouette_scores[k] = silhouette_score(X_scaled, labels)
        except Exception:
            continue

    optimal_k = max(silhouette_scores, key=silhouette_scores.get, default=3)

    # --- Fit final KMeans and assign clusters ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_km = KMeans(n_clusters=optimal_k, n_init=10, random_state=random_state)
        df['cluster'] = final_km.fit_predict(X_scaled)

    # --- Compute centroids in ORIGINAL DATA SPACE (means of features per cluster) ---
    cluster_summary_df = df.groupby('cluster', as_index=False)[features].mean()

    # --- Compute levels per metric from centroid distribution (relative across clusters) ---
    n_clusters = cluster_summary_df.shape[0]
    for feat in features:
        centroid_vals = cluster_summary_df[feat].values
        cluster_summary_df[f'{feat}_level'] = _centroid_level_from_percentiles(centroid_vals, n_clusters)

    # --- Build human-readable cluster_label from the three metric levels ---
    cluster_summary_df['cluster_label'] = (
        cluster_summary_df['hitrate_level'] + " Hitrate & " +
        cluster_summary_df['avg_overpayment_level'] + " Overpayment & " +
        cluster_summary_df['volume_level'] + " Volume"
    )

    # --- Merge labels back to DRG-level dataframe (guarantees alignment) ---
    df = df.merge(cluster_summary_df[['cluster', 'cluster_label']], on='cluster', how='left')

    # --- Optional action rules ---
    if include_action:
        def action_from_levels(row_label):
            # Strict default rules:
            if all(x in row_label for x in ["High Hitrate", "High Overpayment", "High Volume"]):
                return "Include More"
            if all(x in row_label for x in ["Low Hitrate", "Low Overpayment", "Low Volume"]):
                return "Exclude"
            return "Review"

        df['action'] = df['cluster_label'].apply(action_from_levels)

    # --- Final tidy outputs ---
    out_cols = ['drg_code', 'cluster', 'cluster_label']
    if include_action:
        out_cols.append('action')
    drg_cluster_df = df[out_cols].sort_values(['cluster', 'drg_code']).reset_index(drop=True)

    # tidy cluster summary: include levels and original means
    cluster_summary_out = cluster_summary_df[['cluster', 'cluster_label'] + features].copy()

    if return_centroids:
        return drg_cluster_df, cluster_summary_out
    return drg_cluster_df


# ---------------- Example usage ----------------
if __name__ == "__main__":
    sample = pd.DataFrame({
        'drg_code': ['drg_100', 'drg_101', 'drg_102', 'drg_103', 'drg_104', 'drg_105', 'drg_106'],
        'hitrate': [0.70, 0.55, 0.25, 0.88, 0.20, 0.68, 0.33],
        'avg_overpayment': [5800, 3200, 1100, 7200, 800, 4500, 2000],
        'volume': [45, 120, 300, 25, 400, 95, 60]
    })

    drg_map, summary = classify_drg_clusters(sample, return_centroids=True)
    print("\n=== DRG → Cluster Mapping ===")
    print(drg_map.to_string(index=False))
    print("\n=== Cluster Summary ===")
    print(summary.to_string(index=False))
