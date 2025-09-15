import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def cluster_and_explain_drg(df, features):
    """
    This function preprocesses the data, performs K-Means clustering,
    and returns an explained analysis of the resulting clusters.

    Args:
        df (pd.DataFrame): The input DataFrame containing DRG data.
        features (list): A list of column names to be used for clustering.

    Returns:
        pd.DataFrame: The original DataFrame with an added 'Cluster' column.
        pd.DataFrame: A DataFrame with the cluster profiles (centroids).
        pd.DataFrame: A DataFrame with feature importances for clustering.
    """
    print("--- Starting the Clustering Process ---")

    # --- 1. Outlier Removal (using IQR method) ---
    data_to_process = df[features].copy()
    Q1 = data_to_process.quantile(0.25)
    Q3 = data_to_process.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds for outlier detection
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out outliers
    initial_rows = len(data_to_process)
    data_no_outliers = data_to_process[~((data_to_process < lower_bound) | (data_to_process > upper_bound)).any(axis=1)]
    rows_removed = initial_rows - len(data_no_outliers)
    print(f"Removed {rows_removed} rows identified as outliers.")

    # --- 2. Data Standardization ---
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_no_outliers)
    print("Data has been standardized (Mean=0, Std=1).")
    
    # --- 3. Find Optimal Number of Clusters (Elbow Method) ---
    wcss = [] # Within-Cluster Sum of Squares
    k_range = range(1, 11)
    for i in k_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init='auto')
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)
        
    # Plotting the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.show()

    # We typically choose the 'elbow' point. Let's ask the user or automate it.
    optimal_k = int(input("Please enter the optimal number of clusters (K) based on the elbow plot: "))
    
    # --- 4. Perform K-Means Clustering ---
    print(f"\nPerforming K-Means with K={optimal_k}...")
    kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init='auto')
    clusters = kmeans_final.fit_predict(data_scaled)
    
    # Add cluster labels to the non-outlier data
    data_no_outliers['Cluster'] = clusters
    
    # Map clusters back to the original DataFrame
    df_clustered = df.copy()
    df_clustered['Cluster'] = data_no_outliers['Cluster'] # This will be NaN for outliers

    print(f"Clustering complete. Assigned {len(df_clustered.dropna())} DRGs to {optimal_k} clusters.")

    # --- 5. Explainability ---
    
    # a) Cluster Profiles (Centroids)
    centroids_scaled = kmeans_final.cluster_centers_
    # Inverse transform to get centroids in the original scale
    centroids_original = scaler.inverse_transform(centroids_scaled)
    cluster_profiles = pd.DataFrame(centroids_original, columns=features)
    cluster_profiles.index.name = 'Cluster'
    
    # Add cluster size to the profile
    cluster_profiles['Size'] = data_no_outliers['Cluster'].value_counts()

    print("\n--- Cluster Profiles (Centroids in Original Scale) ---")
    print(cluster_profiles)
    
    # b) Feature Importance using a Random Forest
    X = data_no_outliers[features]
    y = data_no_outliers['Cluster']
    
    # Train a classifier to predict the cluster based on the features
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    feature_importances = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\n--- Feature Importance for Cluster Separation ---")
    print(feature_importances)

    # Visualize feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
    plt.title('Feature Importance for Clustering')
    plt.tight_layout()
    plt.show()

    return df_clustered, cluster_profiles, feature_importances

# --- Example Usage ---
if __name__ == '__main__':
    # 1. Create a sample DataFrame (replace this with your actual data)
    # For example: df = pd.read_csv('your_drg_data.csv')
    np.random.seed(42)
    data = {
        'drg_code': [f'DRG_{i}' for i in range(100)],
        'Avg Op': np.random.normal(120, 40, 100).clip(20),
        'Median probability score': np.random.uniform(0.6, 1.0, 100),
        'claim volume': np.random.lognormal(8, 1.5, 100).astype(int) + 100,
        'hitrate': np.random.uniform(0.7, 0.99, 100)
    }
    # Add some outliers for demonstration
    data['claim volume'][5] = 8000
    data['Avg Op'][10] = 500
    
    drg_df = pd.DataFrame(data)

    print("--- Sample Data Head ---")
    print(drg_df.head())
    print("\n")

    # 2. Define the features for clustering
    clustering_features = ['Avg Op', 'Median probability score', 'claim volume', 'hitrate']

    # 3. Run the clustering and explanation function
    clustered_df, profiles, importances = cluster_and_explain_drg(drg_df, clustering_features)
    
    # 4. Inspect the final DataFrame with cluster assignments
    print("\n--- Final DataFrame with Cluster Labels ---")
    print(clustered_df.head(10))
    print("\n--- DRGs per cluster ---")
    print(clustered_df['Cluster'].value_counts())

