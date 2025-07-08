import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

def generate_ae_features_from_diag(df_claims, diag_col='diag_concat', ae_bottleneck_dim=64):
    """
    Generates autoencoder components (features) from a DataFrame
    containing claim IDs and concatenated diagnosis codes.

    Args:
        df_claims (pd.DataFrame): Input DataFrame with 'claim_id' and a column
                                  specified by `diag_col`. This column should contain
                                  a string of comma-separated diagnosis codes (e.g., "I10.9,E11.9").
        diag_col (str): The name of the column containing concatenated diagnosis codes.
                        This column is expected to contain the *final* set of diagnosis codes
                        for each claim, representing the outcome after any 'switching' or 'removal'
                        processes in the medical record.
        ae_bottleneck_dim (int): The desired dimensionality of the autoencoder's
                                 bottleneck layer (number of AE components).

    Returns:
        pd.DataFrame: A DataFrame with 'claim_id' and the generated AE components.
                      Returns None if no unique codes are found or AE training fails.
    """

    print(f"Step 1: Parse and Collect Unique Diagnosis Codes from '{diag_col}'")
    # Split the concatenated string into lists of codes
    # Handle potential NaNs or empty strings in diag_col
    df_claims['parsed_codes'] = df_claims[diag_col].apply(
        lambda x: [code.strip() for code in str(x).split(',') if code.strip()] if pd.notna(x) and x.strip() else []
    )

    # Get all unique codes to build the vocabulary
    all_codes = set()
    for codes_list in df_claims['parsed_codes']:
        all_codes.update(codes_list)

    if not all_codes:
        print(f"Error: No unique diagnosis codes found in the '{diag_col}' column.")
        return None

    # Sort codes for consistent indexing
    vocabulary = sorted(list(all_codes))
    vocab_size = len(vocabulary)
    print(f"Total unique diagnosis codes (vocabulary size): {vocab_size}")

    print("Step 2: Create Multi-Hot Encoded Input for Diagnosis Codes")
    # MultiLabelBinarizer is perfect for converting lists of labels into multi-hot arrays
    mlb = MultiLabelBinarizer(classes=vocabulary) # Ensure consistent column order

    # Transform the parsed codes into multi-hot encoded format
    X_multi_hot = mlb.fit_transform(df_claims['parsed_codes'])
    print(f"Shape of multi-hot encoded diagnosis data: {X_multi_hot.shape}")

    print("Step 3: Design and Train the Autoencoder for Diagnosis Codes")
    # Define the Autoencoder architecture
    input_dim = vocab_size
    encoding_dim = ae_bottleneck_dim # The size of our AE components

    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim * 2, activation='relu')(input_layer) # Example hidden layer
    encoder_output = Dense(encoding_dim, activation='relu', name='bottleneck_layer')(encoder)

    # Decoder
    decoder = Dense(encoding_dim * 2, activation='relu')(encoder_output) # Example hidden layer
    decoder_output = Dense(input_dim, activation='sigmoid')(decoder) # Sigmoid for binary output

    # Full autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoder_output)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

    print("Training autoencoder...")
    history = autoencoder.fit(
        X_multi_hot, X_multi_hot,
        epochs=50,          # Adjust based on data size and convergence
        batch_size=256,     # Adjust based on memory and data size
        shuffle=True,
        verbose=0           # Set to 1 or 2 for training progress
    )
    print("Autoencoder training complete.")
    # print(f"Final Autoencoder Reconstruction Loss: {history.history['loss'][-1]:.4f}")


    print("Step 4: Extract the AE Components (Encoder Output from Diagnosis AE)")
    # Create a separate model for the encoder part
    encoder_model = Model(inputs=input_layer, outputs=encoder_output)

    # Get the AE components for each claim
    ae_components = encoder_model.predict(X_multi_hot)
    print(f"Shape of extracted AE components: {ae_components.shape}")

    print("Step 5: Prepare Output DataFrame")
    # Create column names for the AE components
    ae_column_names = [f'ae_diag_comp_{i+1}' for i in range(ae_bottleneck_dim)] # Renamed for clarity

    # Create the output DataFrame
    df_ae_features = pd.DataFrame(ae_components, columns=ae_column_names)
    df_ae_features['claim_id'] = df_claims['claim_id'].reset_index(drop=True) # Ensure claim_id matches rows

    # Reorder columns to have claim_id first
    df_ae_features = df_ae_features[['claim_id'] + ae_column_names]

    print("AE feature generation for diagnosis codes complete.")
    return df_ae_features

# --- Example Usage ---
if __name__ == "__main__":
    # Sample Data with Diagnosis Codes
    data = {
        'claim_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
        'diag_concat': [ # Using sample ICD-10 like codes
            "I10.9,E11.9",
            "I10.9,J45.9,R51",
            "J45.9,M54.5",
            "E11.9,N18.6",
            "I10.9,E11.9", # Duplicate pattern
            "M54.5,J45.9",
            "I10.9",
            "J45.9,N18.6",
            "E11.9,M54.5,I10.9",
            "M54.5,E11.9,J45.9"
        ]
    }
    df_sample_claims = pd.DataFrame(data)

    # Generate AE features from diagnosis codes with a bottleneck of 8 dimensions
    ae_features_diag_df = generate_ae_features_from_diag(df_sample_claims, ae_bottleneck_dim=8)

    if ae_features_diag_df is not None:
        print("\nResulting AE Features DataFrame (first 5 rows) from Diagnosis Codes:")
        print(ae_features_diag_df.head())
        print(f"\nShape of the resulting DataFrame: {ae_features_diag_df.shape}")

    # Example with a claim having no codes (empty string or NaN)
    data_with_empty_diag = {
        'claim_id': [2001, 2002, 2003],
        'diag_concat': [
            "I10.9,E11.9",
            "", # Empty string
            np.nan # NaN
        ]
    }
    df_empty_diag_claims = pd.DataFrame(data_with_empty_diag)
    print("\n--- Testing with empty/NaN diag_concat ---")
    ae_features_empty_diag_df = generate_ae_features_from_diag(df_empty_diag_claims, ae_bottleneck_dim=4)
    if ae_features_empty_diag_df is not None:
        print("\nResulting AE Features DataFrame (with empty/NaN diag_concat):")
        print(ae_features_empty_diag_df)

    # Example with no unique codes overall
    data_no_diag_codes = {
        'claim_id': [3001, 3002],
        'diag_concat': ["", np.nan]
    }
    df_no_diag_codes = pd.DataFrame(data_no_diag_codes)
    print("\n--- Testing with no unique diagnosis codes overall ---")
    ae_features_no_diag_codes_df = generate_ae_features_from_diag(df_no_diag_codes, ae_bottleneck_dim=2)
