import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

# --- 1. Data Setup (Concatenated Diagnosis Codes with Claim ID) ---
# Your input DataFrame structure now includes 'claim_id'
# REPLACE THIS WITH YOUR ACTUAL INPUT DATAFRAME
data_concatenated = {
    'claim_id': [101, 102, 103, 104, 105, 106],
    'diag':     ['A-B-C', 'D-E-F', 'A-D', 'G-H-I', 'X-Y', 'Z-K'] # Z and K intentionally missing embeddings
}
concatenated_df = pd.DataFrame(data_concatenated)

print("Concatenated DataFrame Head (with Claim ID):")
print(concatenated_df.head())

# --- 2. Embeddings Data (from your dictionary - adjusted for array[[...]] format) ---
# This is a dummy example of your embeddings dictionary with array[[...]] format.
# REPLACE THIS WITH YOUR ACTUAL EMBEDDINGS DICTIONARY
embeddings_dict = {
    'A': np.array([np.random.rand(50)]), # 50 dimensions for diag code 'A'
    'B': np.array([np.random.rand(50)]),
    'C': np.array([np.random.rand(50)]),
    'D': np.array([np.random.rand(50)]),
    'E': np.array([np.random.rand(50)]),
    'F': np.array([np.random.rand(50)]),
    'G': np.array([np.random.rand(50)]),
    'H': np.array([np.random.rand(50)]),
    'I': np.array([np.random.rand(50)]),
    'X': np.array([np.random.rand(50)]),
    'Y': np.array([np.random.rand(50)]),
    # 'Z' and 'K' are intentionally missing to test handling of codes not in dictionary
}

# Convert the embeddings dictionary into a DataFrame
embeddings_list = []
for code, dims_array in embeddings_dict.items():
    row_dict = {'diag_code': code}
    # Flatten the 2D array to a 1D array before iterating over its dimensions
    dims_flat = dims_array.flatten()
    for i, dim_val in enumerate(dims_flat):
        row_dict[f'dim{i+1}'] = dim_val # Assuming dimensions are 1-indexed (dim1 to dim50)
    embeddings_list.append(row_dict)

embeddings_df = pd.DataFrame(embeddings_list)

print("\nEmbeddings DataFrame Head (created from dictionary with array[[...]] format):")
print(embeddings_df.head())
print(f"Shape of Embeddings DataFrame: {embeddings_df.shape}")


# --- 3. Match Embeddings to Concatenated Codes ---

# Explode the 'diag' column in concatenated_df to get individual diagnosis codes
# We need to keep 'claim_id' linked to each individual diagnosis code
exploded_df = concatenated_df.assign(individual_diag=concatenated_df['diag'].apply(lambda x: x.split('-'))).explode('individual_diag')

# Merge the exploded_df with the embeddings_df
# Use 'left' merge to keep all individual diagnosis codes from your concatenated list.
# Codes without a match in embeddings_df will have NaN for their embedding dimensions.
merged_df = pd.merge(exploded_df, embeddings_df, left_on='individual_diag', right_on='diag_code', how='left')

# Drop the original 'diag' and 'diag_code' columns as they are no longer needed
merged_df = merged_df.drop(columns=['diag', 'diag_code'])

# Identify and report diagnosis codes that don't have embeddings
embedding_columns = [f'dim{i}' for i in range(1, 51)] # Assuming 50 dimensions

# Find rows where any embedding dimension is NaN
missing_embedding_rows = merged_df[merged_df[embedding_columns].isnull().any(axis=1)]
missing_diag_codes = missing_embedding_rows['individual_diag'].unique()

if len(missing_diag_codes) > 0:
    print(f"\n--- Warning: Missing Embeddings for the following diagnosis codes ---")
    print(f"These codes were found in your concatenated list but not in your embeddings dictionary:")
    for code in missing_diag_codes:
        print(f"- {code}")
    print(f"These codes will be excluded from PCA. (and thus not contribute to the claim-level PCA result for affected claims)")
else:
    print("\nNo missing embeddings found for any diagnosis codes. All codes will be included in PCA.")

# Prepare data for PCA: Select only rows with complete embeddings
# We need to retain 'claim_id' and 'individual_diag' to link back later
df_for_pca_prep = merged_df[['claim_id', 'individual_diag'] + embedding_columns].copy()
embeddings_with_id_for_pca = df_for_pca_prep.dropna(subset=embedding_columns) # Drop rows where embedding dims are NaN

initial_rows_for_pca = df_for_pca_prep.shape[0]
rows_after_dropping_nan = embeddings_with_id_for_pca.shape[0]

if initial_rows_for_pca > rows_after_dropping_nan:
    print(f"\n{initial_rows_for_pca - rows_after_dropping_nan} out of {initial_rows_for_pca} individual diagnosis codes were dropped due to missing embeddings before PCA.")


# Separate the claim_id, individual_diag, and embedding values for PCA
claim_id_for_pca_mapping = embeddings_with_id_for_pca['claim_id']
individual_diag_for_pca = embeddings_with_id_for_pca['individual_diag']
embedding_values_for_pca = embeddings_with_id_for_pca[embedding_columns]


# --- 4. Perform PCA ---
if embedding_values_for_pca.empty:
    print("\nError: No complete embeddings found for PCA after dropping missing values. Cannot perform PCA.")
else:
    # Ensure n_components is not greater than the number of samples or features
    n_samples, n_features = embedding_values_for_pca.shape
    n_components = min(n_samples, n_features)

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(embedding_values_for_pca)

    # Create a DataFrame for principal components
    pc_column_names = [f'PC{i}' for i in range(1, n_components + 1)]
    pc_df = pd.DataFrame(data=principal_components, columns=pc_column_names)

    # Add the claim_id and individual_diag codes back to the PCA results DataFrame
    pc_df.insert(0, 'individual_diag', individual_diag_for_pca.reset_index(drop=True))
    pc_df.insert(0, 'claim_id', claim_id_for_pca_mapping.reset_index(drop=True))


    # --- 5. Identify the Best Principal Components ---
    # This step is still performed on individual diagnosis codes to understand variance contribution
    explained_variance_ratio = pca.explained_variance_ratio_
    print("\nExplained variance ratio for each principal component (based on individual diag codes):")
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"PC{i+1}: {ratio:.4f}")

    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    print("\nCumulative explained variance (based on individual diag codes):")
    for i, cum_ratio in enumerate(cumulative_explained_variance):
        print(f"PC{i+1}: {cum_ratio:.4f}")

    target_variance = 0.95
    if cumulative_explained_variance[-1] < target_variance:
        print(f"\nWarning: Even with all components, only {cumulative_explained_variance[-1]*100:.2f}% of variance is explained. Target of {target_variance*100}% not reached.")
        num_components_for_target = n_components
    else:
        num_components_for_target = np.where(cumulative_explained_variance >= target_variance)[0][0] + 1

    print(f"\nNumber of components to explain at least {target_variance*100}% variance: {num_components_for_target}")

    # --- 6. Aggregate PCA Results to Claim ID Level ---
    # Group by 'claim_id' and take the mean of the principal components
    # This will give you one row per claim_id, representing the average embedding in the PCA space
    claim_level_pca_df = pc_df.groupby('claim_id')[pc_column_names].mean().reset_index()

    print("\nClaim-level PCA Results Head:")
    print(claim_level_pca_df.head())
    print(f"Shape of Claim-level PCA Results: {claim_level_pca_df.shape}")

    # Save the aggregated principal components to a CSV file
    claim_level_pca_df.to_csv('claim_level_principal_components.csv', index=False)

    print("\nClaim-level principal components calculated and saved to 'claim_level_principal_components.csv'")

