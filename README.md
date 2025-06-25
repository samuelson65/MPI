import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

# Create dummy data for the first table (concatenated diagnosis codes)
data_concatenated = {'diag': ['A-B-C', 'D-E-F', 'A-D', 'G-H-I']}
concatenated_df = pd.DataFrame(data_concatenated)

# Create dummy data for the second table (diagnosis codes and embeddings)
# I'll create embeddings that are easy to distinguish for demonstration
embeddings_data = {
    'diag_code': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
}
# Generate dummy embedding dimensions (dim1 to dim50)
for i in range(1, 51):
    embeddings_data[f'dim{i}'] = np.random.rand(len(embeddings_data['diag_code']))

embeddings_df = pd.DataFrame(embeddings_data)

# Print the head of both dataframes to verify
print("Concatenated DataFrame Head:")
print(concatenated_df.head())
print("\nEmbeddings DataFrame Head:")
print(embeddings_df.head())

# Explode the 'diag' column in concatenated_df to get individual diagnosis codes
# First, split the 'diag' column into a list of codes
concatenated_df['individual_diag'] = concatenated_df['diag'].apply(lambda x: x.split('-'))

# Then, use explode to create a new row for each individual diagnosis code
exploded_df = concatenated_df.explode('individual_diag')

# Merge the exploded_df with the embeddings_df
# The 'individual_diag' from exploded_df should match 'diag_code' from embeddings_df
merged_df = pd.merge(exploded_df, embeddings_df, left_on='individual_diag', right_on='diag_code', how='left')

# Drop the original 'diag' and 'diag_code' columns as they are no longer needed
merged_df = merged_df.drop(columns=['diag', 'diag_code'])

# Handle potential missing embeddings (e.g., if a diagnosis code in concatenated_df has no embedding)
# For now, let's just show how many are missing. A real-world scenario might require imputation or dropping.
print(f"\nNumber of rows with missing embeddings after merge: {merged_df.isnull().sum().sum()}")

# Separate the embeddings for PCA
# The columns from 'dim1' to 'dim50' contain the embeddings
embedding_columns = [f'dim{i}' for i in range(1, 51)]
embeddings_for_pca = merged_df[embedding_columns].dropna() # Drop rows with NaN if any missing after merge

# Perform PCA
n_components = min(embeddings_for_pca.shape[0], embeddings_for_pca.shape[1]) # Choose min of rows or features for n_components
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(embeddings_for_pca)

# Create a DataFrame for principal components
pc_df = pd.DataFrame(data=principal_components, columns=[f'PC{i}' for i in range(1, n_components + 1)])

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
# This is a common way to identify the "best" principal components
target_variance = 0.95
num_components_for_target = np.where(cumulative_explained_variance >= target_variance)[0][0] + 1
print(f"\nNumber of components to explain at least {target_variance*100}% variance: {num_components_for_target}")

# The 'best' principal components are generally those that contribute most to the explained variance,
# or those required to reach a desired cumulative explained variance.
# You can visualize this with a scree plot, but for now, we'll just report.

# Save the principal components to a CSV file
pc_df.to_csv('principal_components.csv', index=False)

print("\nPrincipal components calculated and saved to 'principal_components.csv'")
