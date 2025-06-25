import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

# --- 1. Data Setup (Concatenated Diagnosis Codes) ---
# Assuming you have your concatenated diagnosis codes in a DataFrame like this:
data_concatenated = {'diag': ['A-B-C', 'D-E-F', 'A-D', 'G-H-I', 'X-Y', 'Z']}
concatenated_df = pd.DataFrame(data_concatenated)

print("Concatenated DataFrame Head:")
print(concatenated_df.head())

# --- 2. Embeddings Data (from your dictionary) ---
# This is a dummy example of your embeddings dictionary.
# Replace this with your actual dictionary:
embeddings_dict = {
    'A': np.random.rand(50), # 50 dimensions for diag code 'A'
    'B': np.random.rand(50),
    'C': np.random.rand(50),
    'D': np.random.rand(50),
    'E': np.random.rand(50),
    'F': np.random.rand(50),
    'G': np.random.rand(50),
    'H': np.random.rand(50),
    'I': np.random.rand(50),
    'X': np.random.rand(50), # New codes from X-Y example
    'Y': np.random.rand(50),
    # Note: 'Z' is in concatenated_df but not in embeddings_dict to demonstrate handling missing.
}

# Convert the embeddings dictionary into a DataFrame
# First, prepare a list of dictionaries, one for each diag_code
embeddings_list = []
for code, dims in embeddings_dict.items():
    row_dict = {'diag_code': code}
    for i, dim_val in enumerate(dims):
        row_dict[f'dim{i+1}'] = dim_val # Assuming dimensions are 1-indexed (dim1 to dim50)
    embeddings_list.append(row_dict)

embeddings_df = pd.DataFrame(embeddings_list)

print("\nEmbeddings DataFrame Head (created from dictionary):")
print(embeddings_df.head())
print(f"Shape of Embeddings DataFrame: {embeddings_df.shape}")


# --- 3. Match Embeddings to Concatenated Codes ---

# Explode the 'diag' column in concatenated_df to get individual diagnosis codes
concatenated_df['individual_diag'] = concatenated_df['diag'].apply(lambda x: x.split('-'))
exploded_df = concatenated_df.explode('individual_diag')

# Merge the exploded_df with the embeddings_df
# Use 'left' merge to keep all individual diagnosis codes from your concatenated list,
# even if they don't have a matching embedding (will result in NaNs for embedding dimensions).
merged_df = pd.merge(exploded_df, embeddings_df, left_on='individual_diag', right_on='diag_code', how='left')

# Drop the original 'diag' and 'diag_code' columns as they are no longer needed
merged_df = merged_df.drop(columns=['diag', 'diag_code'])

# Handle potential missing embeddings (e.g., if a diagnosis code in concatenated_df has no embedding)
# If a diagnosis code was in 'concatenated_df' but not in 'embeddings_dict', its embedding dimensions will be NaN.
# PCA cannot handle NaNs, so we need to decide how to handle them.
# Common approaches:
# 1. Drop rows with any missing embeddings (simple, but loses data)
# 2. Impute missing values (e.g., with mean, median, or a specific value)
# For this script, we'll drop them for simplicity, but consider imputation for larger datasets.
initial_rows = merged_df.shape[0]
embedding_columns = [f'dim{i}' for i in range(1, 51)] # Assuming 50 dimensions
embeddings_for_pca = merged_df[embedding_columns].copy() # Create a copy to avoid SettingWithCopyWarning
embeddings_for_pca.dropna(inplace=True) # Drop rows where any embedding dimension is NaN

rows_after_dropping_nan = embeddings_for_pca.shape[0]
print(f"\nTotal individual diagnosis codes before dropping NaNs: {initial_rows}")
print(f"Total individual diagnosis codes after dropping NaNs: {rows_after_dropping_nan}")
if initial_rows > rows_after_dropping_nan:
    print(f"Warning: {initial_rows - rows_after_dropping_nan} rows were dropped due to missing embeddings.")
    print("Consider imputing missing values if data loss is a concern.")


# --- 4. Perform PCA ---
if embeddings_for_pca.empty:
    print("\nError: No complete embeddings found for PCA after dropping missing values. Cannot perform PCA.")
else:
    # Ensure n_components is not greater than the number of samples or features
    n_samples, n_features = embeddings_for_pca.shape
    n_components = min(n_samples, n_features)

    # If all values are identical for a dimension, variance will be 0, which can cause issues.
    # It's good practice to scale data before PCA, especially if dimensions have different scales or ranges,
    # though with embeddings, they are often already in a normalized space.
    # For simplicity, we're skipping explicit scaling here, assuming embeddings are pre-scaled.

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(embeddings_for_pca)

    # Create a DataFrame for principal components
    pc_column_names = [f'PC{i}' for i in range(1, n_components + 1)]
    pc_df = pd.DataFrame(data=principal_components, columns=pc_column_names)

    # --- 5. Identify the Best Principal Components ---

    # Explained variance ratio for each component
    explained_variance_ratio = pca.explained_variance_ratio_
    print("\nExplained variance ratio for each principal component:")
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"PC{i+1}: {ratio:.4f}")

    # Cumulative explained variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    print("\nCumulative explained variance:")
    for i, cum_ratio in enumerate(cumulative_explained_variance):
        print(f"PC{i+1}: {cum_ratio:.4f}")

    # Determine the number of components to explain a certain percentage of variance (e.g., 95%)
    target_variance = 0.95
    if cumulative_explained_variance[-1] < target_variance:
        print(f"\nWarning: Even with all components, only {cumulative_explained_variance[-1]*100:.2f}% of variance is explained. Target of {target_variance*100}% not reached.")
        num_components_for_target = n_components
    else:
        num_components_for_target = np.where(cumulative_explained_variance >= target_variance)[0][0] + 1

    print(f"\nNumber of components to explain at least {target_variance*100}% variance: {num_components_for_target}")

    # Save the principal components to a CSV file
    # Note: The principal_components.csv will contain a row for each diagnosis code
    # that had a complete embedding, not necessarily original 'diag' row.
    pc_df.to_csv('principal_components.csv', index=False)

    print("\nPrincipal components calculated and saved to 'principal_components.csv'")

