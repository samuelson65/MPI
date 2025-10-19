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
    Assigns cluster labels using relative ranking of centroid values instead of global medians.

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

    # --- 5. Compute centroids in original scale ---
    cluster_summary_df = df.groupby('cluster')[num_cols].mean().reset_index()

    # --- 6. Rank clusters on each metric ---
    for col in num_cols:
        # higher value → higher rank
        cluster_summary_df[f'{col}_rank'] = cluster_summary_df[col].rank(ascending=True).astype(int)

    # --- 7. Assign human-readable qualitative labels ---
    def qualitative_label(val, total_clusters):
        if val >= total_clusters - 0.5:
            return "High"
        elif val <= 1.5:
            return "Low"
        else:
            return "Medium"

    total_clusters = cluster_summary_df['cluster'].nunique()
    for col in num_cols:
        cluster_summary_df[f'{col}_level'] = cluster_summary_df[f'{col}_rank'].apply(lambda x: qualitative_label(x, total_clusters))

    # --- 8. Build descriptive label ---
    cluster_summary_df['cluster_label'] = (
        cluster_summary_df.apply(
            lambda r: f"{r['hitrate_level']} Hitrate & {r['avg_overpayment_level']} Overpayment & {r['volume_level']} Volume", axis=1
        )
    )

    # --- 9. Merge back to main DF ---
    df = df.merge(cluster_summary_df[['cluster', 'cluster_label']], on='cluster', how='left')

    # --- 10. Optional: add action rules ---
    if include_action:
        def action_from_label(lbl):
            if all(x in lbl for x in ["High Hitrate", "High Overpayment", "High Volume"]):
                return "Include More"
            elif all(x in lbl for x in ["Low Hitrate", "Low Overpayment", "Low Volume"]):
                return "Exclude"
            return "Review"

        df['action'] = df['cluster_label'].apply(action_from_label)

    # --- 11. Final outputs ---
    drg_cluster_df = df[['drg_code', 'cluster', 'cluster_label'] + (['action'] if include_action else [])]
    cluster_summary_df = cluster_summary_df[['cluster', 'cluster_label'] + num_cols]

    return (drg_cluster_df, cluster_summary_df) if return_centroids else drg_cluster_df


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    sample = pd.DataFrame({
        'drg_code': ['drg_100', 'drg_101', 'drg_102', 'drg_103', 'drg_104', 'drg_105', 'drg_106'],
        'hitrate': [0.70, 0.55, 0.25, 0.88, 0.20, 0.68, 0.33],
        'avg_overpayment': [5800, 3200, 1100, 7200, 800, 4500, 2000],
        'volume': [45, 120, 300, 25, 400, 95, 60]
    })

    drg_map, summary = classify_drg_clusters(sample)

    print("\n=== DRG → Cluster Mapping ===")
    print(drg_map)

    print("\n=== Cluster Summary ===")
    print(summary)
