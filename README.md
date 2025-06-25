import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler # Good practice for neural networks

# --- 1. Data Setup (Concatenated Diagnosis Codes with Claim ID) ---
# REPLACE THIS WITH YOUR ACTUAL INPUT DATAFRAME
data_concatenated = {
    'claim_id': [101, 102, 103, 104, 105, 106, 107, 108], # Added more claims for better AE training
    'diag':     ['A-B-C', 'D-E-F', 'A-D', 'G-H-I', 'X-Y', 'Z-K', 'A-B-X', 'C-D-Y']
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
    'Q': np.array([np.random.rand(50)]), # Add some more for more diverse training data
    'R': np.array([np.random.rand(50)]),
    'S': np.array([np.random.rand(50)]),
    # 'Z' and 'K' are intentionally missing to test handling of codes not in dictionary
}

# Convert the embeddings dictionary into a DataFrame
embeddings_list = []
for code, dims_array in embeddings_dict.items():
    row_dict = {'diag_code': code}
    dims_flat = dims_array.flatten()
    if len(dims_flat) != 50: # Basic check for correct embedding dimension
        print(f"Warning: Diagnosis code '{code}' does not have 50 dimensions. Skipping.")
        continue
    for i, dim_val in enumerate(dims_flat):
        row_dict[f'dim{i+1}'] = dim_val
    embeddings_list.append(row_dict)

embeddings_df = pd.DataFrame(embeddings_list)

print("\nEmbeddings DataFrame Head (created from dictionary with array[[...]] format):")
print(embeddings_df.head())
print(f"Shape of Embeddings DataFrame: {embeddings_df.shape}")


# --- 3. Match Embeddings to Concatenated Codes ---

# Explode the 'diag' column in concatenated_df to get individual diagnosis codes
exploded_df = concatenated_df.assign(individual_diag=concatenated_df['diag'].apply(lambda x: x.split('-'))).explode('individual_diag')

# Merge the exploded_df with the embeddings_df
merged_df = pd.merge(exploded_df, embeddings_df, left_on='individual_diag', right_on='diag_code', how='left')

# Drop the original 'diag' and 'diag_code' columns
merged_df = merged_df.drop(columns=['diag', 'diag_code'])

# Identify and report diagnosis codes that don't have embeddings
embedding_columns = [f'dim{i}' for i in range(1, 51)] # Assuming 50 dimensions

missing_embedding_rows = merged_df[merged_df[embedding_columns].isnull().any(axis=1)]
missing_diag_codes = missing_embedding_rows['individual_diag'].unique()

if len(missing_diag_codes) > 0:
    print(f"\n--- Warning: Missing Embeddings for the following diagnosis codes ---")
    print(f"These codes were found in your concatenated list but not in your embeddings dictionary:")
    for code in missing_diag_codes:
        print(f"- {code}")
    print(f"These codes will be excluded from Autoencoder training and feature extraction (and thus not contribute to the claim-level result for affected claims).")
else:
    print("\nNo missing embeddings found for any diagnosis codes. All codes will be included.")

# Prepare data for Autoencoder: Select only rows with complete embeddings
df_for_ae_prep = merged_df[['claim_id', 'individual_diag'] + embedding_columns].copy()
embeddings_with_id_for_ae = df_for_ae_prep.dropna(subset=embedding_columns) # Drop rows where embedding dims are NaN

initial_rows_for_ae = df_for_ae_prep.shape[0]
rows_after_dropping_nan = embeddings_with_id_for_ae.shape[0]

if initial_rows_for_ae > rows_after_dropping_nan:
    print(f"\n{initial_rows_for_ae - rows_after_dropping_nan} out of {initial_rows_for_ae} individual diagnosis codes were dropped due to missing embeddings before Autoencoder processing.")


# Separate the claim_id, individual_diag, and embedding values for Autoencoder
claim_id_for_ae_mapping = embeddings_with_id_for_ae['claim_id'].reset_index(drop=True)
individual_diag_for_ae = embeddings_with_id_for_ae['individual_diag'].reset_index(drop=True)
embedding_values_for_ae = embeddings_with_id_for_ae[embedding_columns]


# --- 4. Prepare and Scale Data for Autoencoder ---
# It's good practice to scale data for neural networks
scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(embedding_values_for_ae)

# Check if there's enough data to train
if scaled_embeddings.shape[0] < 5: # Arbitrary small number, adjust as needed
    print("\nError: Not enough complete embeddings to train an Autoencoder. Need at least 5 unique embeddings after dropping NaNs.")
    exit() # Exit the script if not enough data


# --- 5. Build and Train the Autoencoder ---
input_dim = 50 # Original embedding dimension
latent_dim = 10 # This is the number of reduced dimensions (your "principal components" from AE) - YOU CAN TUNE THIS!

# Encoder
input_layer = Input(shape=(input_dim,))
encoder_layer = Dense(30, activation='relu')(input_layer) # Hidden layer 1
latent_layer = Dense(latent_dim, activation='relu', name='latent_space')(encoder_layer) # Bottleneck layer

# Decoder
decoder_layer = Dense(30, activation='relu')(latent_layer) # Hidden layer 2
output_layer = Dense(input_dim, activation='linear')(decoder_layer) # Output layer matches input dim

# Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse') # Mean Squared Error is common for reconstruction

print("\nAutoencoder Model Summary:")
autoencoder.summary()

# Train the autoencoder
print(f"\nTraining Autoencoder with {scaled_embeddings.shape[0]} samples...")
history = autoencoder.fit(scaled_embeddings, scaled_embeddings,
                          epochs=50, # Number of training iterations - TUNE THIS!
                          batch_size=32, # Number of samples per gradient update - TUNE THIS!
                          shuffle=True,
                          verbose=0 # Set to 1 for progress bar during training
                         )

print(f"Autoencoder training finished. Final Loss: {history.history['loss'][-1]:.4f}")


# --- 6. Extract Encoded Features (Latent Space) ---
# Create an encoder model that outputs the latent space representation
encoder = Model(inputs=input_layer, outputs=latent_layer)
encoded_embeddings = encoder.predict(scaled_embeddings)

# Create a DataFrame for the encoded features
ae_column_names = [f'AE_PC{i}' for i in range(1, latent_dim + 1)] # Naming them AE_PC to differentiate
ae_df = pd.DataFrame(data=encoded_embeddings, columns=ae_column_names)

# Add claim_id and individual_diag back to the AE features DataFrame
ae_df.insert(0, 'individual_diag', individual_diag_for_ae)
ae_df.insert(0, 'claim_id', claim_id_for_ae_mapping)

print("\nHead of Autoencoder-generated features (individual diag code level):")
print(ae_df.head())


# --- 7. Aggregate Autoencoder Features to Claim ID Level ---
# Group by 'claim_id' and take the mean of the Autoencoder features
claim_level_ae_df = ae_df.groupby('claim_id')[ae_column_names].mean().reset_index()

print("\nClaim-level Autoencoder Features Head:")
print(claim_level_ae_df.head())
print(f"Shape of Claim-level Autoencoder Features: {claim_level_ae_df.shape}")

# Save the aggregated Autoencoder features to a CSV file
claim_level_ae_df.to_csv('claim_level_autoencoder_features.csv', index=False)

print("\nClaim-level Autoencoder features calculated and saved to 'claim_level_autoencoder_features.csv'")

