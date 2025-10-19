import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.exceptions import ConvergenceWarning
import warnings

def drg_cluster_analysis(df: pd.DataFrame,
                         feature_cols=None,
                         k_min=2,
                         k_max=8,
                         random_state=42,
                         verbose=True):
    """
    Cluster DRG codes based on model and financial metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with DRG code and metric columns.
    feature_cols : list, optional
        List of numeric columns (in lowercase). 
        Defaults: ['median_probability', 'hitrate', 'avg_overpayment', 'volume'].
    k_min, k_max : int
        Range of cluster sizes to test.
    random_state : int
        Seed for reproducibility.
    verbose : bool
        If True, prints progress logs.

    Returns
    -------
    dict with:
        - 'df_clusters': DataFrame with ['drgcode', 'cluster', 'cluster_label']
        - 'summary': DataFrame with mean metrics per cluster
        - 'model': trained KMeans object
        - 'scaler': trained StandardScaler
    """

    # --- Set defaults ---
    if feature_cols is None:
        feature_cols = ['median_probability', 'hitrate', 'avg_overpayment', 'volume']

    # --- Validate required columns ---
    required = ['drgcode'] + feature_cols
    missing = [c for c in required if c not in df.columns.str.lower()]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- Standardize column names ---
    df = df.rename(columns={c: c.lower() for c in df.columns})

    # --- Drop invalid rows ---
    df = df.dropna(subset=feature_cols)
    if df.empty:
        raise ValueError("No valid rows left after dropping NaNs.")

    # Replace inf values
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # --- Handle small dataset edge case ---
    n_samples = df.shape[0]
    if n_samples < k_min:
        raise ValueError(f"Too few DRGs ({n_samples}) for clustering (need ≥ {k_min}).")

    # --- Scale features ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    # --- Find optimal number of clusters ---
    silhouette_scores = {}
    best_k, best_score = k_min, -1

    for k in range(k_min, min(k_max, n_samples) + 1):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
                labels = km.fit_predict(X_scaled)
                if len(set(labels)) > 1:
                    score = silhouette_score(X_scaled, labels)
                    silhouette_scores[k] = score
                    if score > best_score:
                        best_k, best_score = k, score
        except Exception as e:
            if verbose:
                print(f"⚠️ Skipped k={k}: {e}")

    if verbose and silhouette_scores:
        print("Silhouette scores:", silhouette_scores)

    # --- Fit final model ---
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # --- Compute cluster summary with means ---
    summary = (
        df.groupby('cluster')[feature_cols]
        .mean()
        .round(3)
        .reset_index()
    )

    # --- Derive human-readable labels ---
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=feature_cols)
    cluster_labels = []
    for _, row in centroids.iterrows():
        if row['median_probability'] > 0.7 and row['hitrate'] > 0.6:
            cluster_labels.append("high-confidence")
        elif row['avg_overpayment'] > np.median(centroids['avg_overpayment']):
            cluster_labels.append("high-risk")
        elif row['volume'] > np.median(centroids['volume']):
            cluster_labels.append("high-volume")
        else:
            cluster_labels.append("moderate")

    label_map = dict(enumerate(cluster_labels))
    df['cluster_label'] = df['cluster'].map(label_map)

    # --- Output for downstream modeling ---
    df_clusters = df[['drgcode', 'cluster', 'cluster_label']].copy()

    if verbose:
        print(f"\n✅ Optimal clusters: {best_k} (Silhouette: {best_score:.3f})")
        print("\nCluster Summary:\n", summary)
        print("\nLabel Mapping:\n", label_map)

    return {
        'df_clusters': df_clusters,
        'summary': summary,
        'model': kmeans,
        'scaler': scaler
    }
