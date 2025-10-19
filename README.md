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
    Cluster DRG codes based on probability, hitrate, overpayment, and volume.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with DRGCode and metric columns.
    feature_cols : list, optional
        List of numeric columns to use for clustering. Defaults to
        ['Median_Probability', 'HitRate', 'Avg_Overpayment', 'Volume'].
    k_min, k_max : int
        Range of cluster sizes to test for optimal k.
    random_state : int
        Seed for reproducibility.
    verbose : bool
        Whether to print progress and summaries.

    Returns
    -------
    dict with:
        - 'df': Original dataframe with cluster labels
        - 'summary': Cluster-level mean metrics
        - 'model': Fitted KMeans model
        - 'scaler': Fitted StandardScaler
    """

    if feature_cols is None:
        feature_cols = ['Median_Probability', 'HitRate', 'Avg_Overpayment', 'Volume']

    # --- Validate input ---
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()

    # --- Handle missing or invalid data ---
    df = df.dropna(subset=feature_cols)
    if df.empty:
        raise ValueError("No valid rows left after dropping NaNs.")

    # Replace inf or -inf
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # --- Handle small dataset edge case ---
    n_samples = df.shape[0]
    if n_samples < k_min:
        raise ValueError(f"Too few DRGs ({n_samples}) for clustering (need ≥ {k_min}).")

    # --- Scale features ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    # --- Optimal K search with safety ---
    silhouette_scores = {}
    best_k = k_min
    best_score = -1

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
                        best_score = score
                        best_k = k
        except Exception as e:
            if verbose:
                print(f"⚠️ Skipped k={k}: {e}")

    if verbose and silhouette_scores:
        print("Silhouette scores:", silhouette_scores)

    # --- Fit final model ---
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # --- Compute cluster summary ---
    cluster_summary = (
        df.groupby('Cluster')[feature_cols]
        .agg(['mean', 'median', 'count'])
        .round(3)
    )

    # --- Optional human-readable cluster naming ---
    centroids = pd.DataFrame(
        kmeans.cluster_centers_, columns=feature_cols
    ).apply(lambda x: np.round(x, 3))

    cluster_labels = []
    for _, row in centroids.iterrows():
        if row['Median_Probability'] > 0.7 and row['HitRate'] > 0.6:
            cluster_labels.append("High-Confidence")
        elif row['Avg_Overpayment'] > np.median(centroids['Avg_Overpayment']):
            cluster_labels.append("High-Risk")
        elif row['Volume'] > np.median(centroids['Volume']):
            cluster_labels.append("High-Volume")
        else:
            cluster_labels.append("Moderate")

    label_map = dict(enumerate(cluster_labels))
    df['Cluster_Label'] = df['Cluster'].map(label_map)

    # --- Output summary ---
    if verbose:
        print(f"\n✅ Optimal number of clusters: {best_k} (Silhouette: {best_score:.3f})")
        print("\nCluster centroids:\n", centroids)
        print("\nCluster summary:\n", cluster_summary)
        print("\nLabel mapping:\n", label_map)

    return {
        'df': df,
        'summary': cluster_summary,
        'model': kmeans,
        'scaler': scaler
    }

result = drg_cluster_analysis(df, verbose=True)

clustered_df = result['df']
summary_df = result['summary']

# Optional: apply custom thresholds later
threshold_map = {
    "High-Confidence": 0.9,
    "High-Volume": 0.75,
    "Moderate": 0.7,
    "High-Risk": 0.6
}
clustered_df['Adjusted_Threshold'] = clustered_df['Cluster_Label'].map(threshold_map)
