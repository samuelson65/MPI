import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def classify_drg_clusters(df, random_state=42, min_clusters=2, max_clusters=8, return_centroids=False):
    """
    Clusters DRG codes (e.g., 'drg_100') based on performance and financial metrics.

    Parameters:
        df (pd.DataFrame): Must contain columns:
            ['drg_code', 'median_probability', 'hitrate', 'avg_overpayment', 'volume']
        random_state (int): Random seed for reproducibility.
        min_clusters (int): Minimum number of clusters to evaluate.
        max_clusters (int): Maximum number of clusters to evaluate.
        return_centroids (bool): If True, returns both the DRG-cluster mapping and centroid summary.

    Returns:
        pd.DataFrame or (pd.DataFrame, pd.DataFrame):
            - drg_cluster_df: ['drg_code', 'cluster', 'cluster_label']
            - (optional) cluster_summary_df: centroid-level mean metrics
    """

    # --- 1. Validate input ---
    required_cols = ['drg_code', 'median_probability', 'hitrate', 'avg_overpayment', 'volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # --- 2. Ensure proper data types ---
    num_cols = ['median_probability', 'hitrate', 'avg_overpayment', 'volume']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=num_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise ValueError("DataFrame has no valid numeric data after cleaning.")

    # --- 3. Adjust max_clusters for small datasets ---
    max_clusters = min(max_clusters, max(2, len(df) // 2))

    # --- 4. Scale numeric features ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[num_cols])

    # --- 5. Determine optimal number of clusters ---
    silhouette_scores = {}
    for k in range(min_clusters, max_clusters + 1):
        try:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = km.fit_predict(X_scaled)
            silhouette_scores[k] = silhouette_score(X_scaled, labels)
        except Exception:
            continue

    optimal_k = max(silhouette_scores, key=silhouette_scores.get, default=3)

    # --- 6. Final KMeans fit ---
    kmeans = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # --- 7. Compute cluster centroids (in original scale) ---
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=num_cols
    )
    centroids['cluster'] = centroids.index

    # --- 8. Generate descriptive cluster labels ---
    labels = []
    medians = {col: df[col].median() for col in num_cols}
    for _, row in centroids.iterrows():
        desc = []
        desc.append("High Hitrate" if row['hitrate'] > medians['hitrate'] else "Low Hitrate")
        desc.append("High Overpayment" if row['avg_overpayment'] > medians['avg_overpayment'] else "Low Overpayment")
        desc.append("High Volume" if row['volume'] > medians['volume'] else "Low Volume")
        desc.append("High Probability" if row['median_probability'] > medians['median_probability'] else "Low Probability")
        labels.append(" & ".join(desc))
    centroids['cluster_label'] = labels

    # --- 9. Merge back cluster labels to main df ---
    df = df.merge(centroids[['cluster', 'cluster_label']], on='cluster', how='left')

    # --- 10. Final tidy output ---
    drg_cluster_df = df[['drg_code', 'cluster', 'cluster_label']].sort_values('cluster').reset_index(drop=True)
    cluster_summary_df = centroids[['cluster'] + num_cols + ['cluster_label']]

    if return_centroids:
        return drg_cluster_df, cluster_summary_df
    return drg_cluster_df


# ---------------- Example usage ----------------
if __name__ == "__main__":
    data = {
        'drg_code': ['drg_100', 'drg_101', 'drg_102', 'drg_103', 'drg_104', 'drg_105'],
        'median_probability': [0.82, 0.60, 0.45, 0.91, 0.38, 0.75],
        'hitrate': [0.70, 0.55, 0.25, 0.88, 0.20, 0.68],
        'avg_overpayment': [5800, 3200, 1100, 7200, 800, 4500],
        'volume': [45, 120, 300, 25, 400, 95]
    }
    df = pd.DataFrame(data)

    drg_cluster_df, cluster_summary_df = classify_drg_clusters(df, return_centroids=True)

    print("\n=== DRG Cluster Mapping ===")
    print(drg_cluster_df)

    print("\n=== Cluster Summary ===")
    print(cluster_summary_df)
