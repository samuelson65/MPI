import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

# --- 1. Data Setup (Concatenated Diagnosis Codes) ---
# Replace this with your actual concatenated diagnosis codes
data_concatenated = {'diag': ['A-B-C', 'D-E-F', 'A-D', 'G-H-I', 'X-Y', 'Z-K']} # Added 'Z-K' for more missing
concatenated_df = pd.DataFrame(data_concatenated)

print("Concatenated DataFrame Head:")
print(concatenated_df.head())

# --- 2. Embeddings Data (from your dictionary) ---
# This is a dummy example of your embeddings dictionary.
# REPLACE THIS WITH YOUR ACTUAL EMBEDDINGS DICTIONARY
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
    'X': np.random.rand(50),
    'Y': np.random.rand(50),
    # Note: 'Z' and 'K' are in concatenated_df but not in embeddings_dict to demonstrate handling missing.
}

# Convert the embeddings dictionary into a DataFrame
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
    print(f"These codes will be excluded from PCA.")
else:
    print("\nNo missing embeddings found for any diagnosis codes. All codes will be included in PCA.")

# Prepare data for PCA: Select only rows with complete embeddings
# First, create a temporary DataFrame that includes 'individual_diag'
# along with the embedding columns, then drop NaNs from this.
df_for_pca_prep = merged_df[['individual_diag'] + embedding_columns].copy()
embeddings_for_pca = df_for_pca_prep.dropna(subset=embedding_columns) # Drop rows where embedding dims are NaN

initial_rows_for_pca = df_for_pca_prep.shape[0]
rows_after_dropping_nan = embeddings_for_pca.shape[0]

if initial_rows_for_pca > rows_after_dropping_nan:
    print(f"\n{initial_rows_for_pca - rows_after_dropping_nan} out of {initial_rows_for_pca} individual diagnosis codes were dropped due to missing embeddings before PCA.")


# Separate the individual_diag codes that will go into PCA, and the embedding values
individual_diag_for_pca = embeddings_for_pca['individual_diag']
embedding_values_for_pca = embeddings_for_pca[embedding_columns]


# --- 4. Perform PCA ---
if embedding_values_for_pca.empty:
    print("\nError: No complete embeddings found for PCA after dropping missing values. Cannot perform PCA.")
else:
    # Ensure n_components is not greater than the number of samples or features
    n_samples, n_features = embedding_values_for_pca.shape
    n_components = min(n_samples, n_features)

    # It's generally good practice to scale data before PCA, especially if dimensions have different scales.
    # However, embeddings are often already in a normalized space, so scaling might not be strictly necessary
    # depending on how your embeddings were generated. For robustness, you might consider StandardScaler.
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # scaled_embeddings = scaler.fit_transform(embedding_values_for_pca)
    # pca = PCA(n_components=n_components)
    # principal_components = pca.fit_transform(scaled_embeddings)

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(embedding_values_for_pca)

    # Create a DataFrame for principal components
    pc_column_names = [f'PC{i}' for i in range(1, n_components + 1)]
    pc_df = pd.DataFrame(data=principal_components, columns=pc_column_names)

    # Add the individual_diag codes back to the PCA results DataFrame
    pc_df.insert(0, 'individual_diag', individual_diag_for_pca.reset_index(drop=True))

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
    # This CSV will now include the 'individual_diag' column
    pc_df.to_csv('principal_components_with_diag_codes.csv', index=False)

    print("\nPrincipal components calculated and saved to 'principal_components_with_diag_codes.csv'")

