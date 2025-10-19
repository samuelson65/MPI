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
    Cluster DRG codes (e.g. 'drg_100') using hitrate, avg_overpayment, and volume.
    Returns aligned labels and actions per cluster.

    Parameters:
        df (pd.DataFrame): Must contain ['drg_code', 'hitrate', 'avg_overpayment', 'volume']
        random_state (int): Random seed for reproducibility.
        min_clusters (int): Minimum clusters to test.
        max_clusters (int): Maximum clusters to test.
        return_centroids (bool): Whether to return cluster summary.
        include_action (bool): Whether to classify clusters as Include/Exclude/Review.

    Returns:
        drg_cluster_df (pd.DataFrame)
        cluster_summary_df (pd.DataFrame, optional)
    """

    # --- 1. Input validation ---
    required_cols = ['drg_code', 'hitrate', 'avg_overpayment', 'volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df['drg_code'] = df['drg_code'].astype(str)

    # numeric conversion
    num_cols = ['hitrate', 'avg_overpayment', 'volume']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=num_cols).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid data rows after cleaning numeric columns.")

    # adjust cluster range
    max_clusters = min(max_clusters, max(2, len(df) // 2))

    # --- 2. Standardize data ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[num_cols])

    # --- 3. Determine optimal K ---
    silhouette_scores = {}
    for k in range(min_clusters, max_clusters + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                labels = km.fit_predict(X_scaled)
            if len(set(labels)) > 1:
                silhouette_scores[k] = silhouette_score(X_scaled, labels)
        except Exception:
            continue

    optimal_k = max(silhouette_scores, key=silhouette_scores.get, default=3)

    # --- 4. Fit final KMeans ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_km = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
        df['cluster'] = final_km.fit_predict(X_scaled)

    # --- 5. Compute true centroids in ORIGINAL space ---
    cluster_summary_df = df.groupby('cluster')[num_cols].mean().reset_index()

    # --- 6. Assign cluster labels based on median comparisons ---
    medians = df[num_cols].median()

    def make_label(row):
        parts = []
        parts.append("High Hitrate" if row['hitrate'] > medians['hitrate'] else "Low Hitrate")
        parts.append("High Overpayment" if row['avg_overpayment'] > medians['avg_overpayment'] else "Low Overpayment")
        parts.append("High Volume" if row['volume'] > medians['volume'] else "Low Volume")
        return " & ".join(parts)

    cluster_summary_df['cluster_label'] = cluster_summary_df.apply(make_label, axis=1)

    # --- 7. Merge back to main DF (aligned) ---
    df = df.merge(cluster_summary_df[['cluster', 'cluster_label']], on='cluster', how='left')

    # --- 8. Optional action classification ---
    if include_action:
        def action_from_label(lbl):
            if all(x in lbl for x in ["High Hitrate", "High Overpayment", "High Volume"]):
                return "Include More"
            elif all(x in lbl for x in ["Low Hitrate", "Low Overpayment", "Low Volume"]):
                return "Exclude"
            return "Review"

        df['action'] = df['cluster_label'].apply(action_from_label)

    # --- 9. Final outputs ---
    out_cols = ['drg_code', 'cluster', 'cluster_label']
    if include_action:
        out_cols.append('action')

    drg_cluster_df = df[out_cols].sort_values(['cluster', 'drg_code']).reset_index(drop=True)
    cluster_summary_df = cluster_summary_df[['cluster', 'cluster_label'] + num_cols]

    if return_centroids:
        return drg_cluster_df, cluster_summary_df
    else:
        return drg_cluster_df


# ---------------- Example usage ----------------
if __name__ == "__main__":
    sample = pd.DataFrame({
        'drg_code': ['drg_100', 'drg_101', 'drg_102', 'drg_103', 'drg_104', 'drg_105', 'drg_106'],
        'hitrate': [0.70, 0.55, 0.25, 0.88, 0.20, 0.68, 0.33],
        'avg_overpayment': [5800, 3200, 1100, 7200, 800, 4500, 2000],
        'volume': [45, 120, 300, 25, 400, 95, 60]
    })

    drg_map, cluster_summary = classify_drg_clusters(sample, return_centroids=True)

    print("\n=== DRG → Cluster Mapping ===")
    print(drg_map)

    print("\n=== Cluster Summary ===")
    print(cluster_summary)
