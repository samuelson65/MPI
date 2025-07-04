import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

def generate_ae_features(df_claims, proc_col='proc_concat', ae_bottleneck_dim=64):
    """
    Generates autoencoder components (features) from a DataFrame
    containing claim IDs and concatenated procedure codes.

    Args:
        df_claims (pd.DataFrame): Input DataFrame with 'claim_id' and 'proc_concat' columns.
                                  'proc_concat' should contain a string of comma-separated
                                  procedure codes (e.g., "0JB,0HZ").
        proc_col (str): The name of the column containing concatenated procedure codes.
        ae_bottleneck_dim (int): The desired dimensionality of the autoencoder's
                                 bottleneck layer (number of AE components).

    Returns:
        pd.DataFrame: A DataFrame with 'claim_id' and the generated AE components.
                      Returns None if no unique codes are found or AE training fails.
    """

    print("Step 1: Parse and Collect Unique Codes")
    # Split the concatenated string into lists of codes
    # Handle potential NaNs or empty strings in proc_concat
    df_claims['parsed_codes'] = df_claims[proc_col].apply(
        lambda x: [code.strip() for code in str(x).split(',') if code.strip()] if pd.notna(x) and x.strip() else []
    )

    # Get all unique codes to build the vocabulary
    all_codes = set()
    for codes_list in df_claims['parsed_codes']:
        all_codes.update(codes_list)

    if not all_codes:
        print("Error: No unique procedure codes found in the 'proc_concat' column.")
        return None

    # Sort codes for consistent indexing
    vocabulary = sorted(list(all_codes))
    vocab_size = len(vocabulary)
    print(f"Total unique codes (vocabulary size): {vocab_size}")

    print("Step 2: Create Multi-Hot Encoded Input")
    # MultiLabelBinarizer is perfect for converting lists of labels into multi-hot arrays
    mlb = MultiLabelBinarizer(classes=vocabulary) # Ensure consistent column order

    # Transform the parsed codes into multi-hot encoded format
    X_multi_hot = mlb.fit_transform(df_claims['parsed_codes'])
    print(f"Shape of multi-hot encoded data: {X_multi_hot.shape}")

    print("Step 3: Design and Train the Autoencoder")
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


    print("Step 4: Extract the AE Components (Encoder Output)")
    # Create a separate model for the encoder part
    encoder_model = Model(inputs=input_layer, outputs=encoder_output)

    # Get the AE components for each claim
    ae_components = encoder_model.predict(X_multi_hot)
    print(f"Shape of extracted AE components: {ae_components.shape}")

    print("Step 5: Prepare Output DataFrame")
    # Create column names for the AE components
    ae_column_names = [f'ae_comp_{i+1}' for i in range(ae_bottleneck_dim)]

    # Create the output DataFrame
    df_ae_features = pd.DataFrame(ae_components, columns=ae_column_names)
    df_ae_features['claim_id'] = df_claims['claim_id'].reset_index(drop=True) # Ensure claim_id matches rows

    # Reorder columns to have claim_id first
    df_ae_features = df_ae_features[['claim_id'] + ae_column_names]

    print("AE feature generation complete.")
    return df_ae_features

# --- Example Usage ---
if __name__ == "__main__":
    # Sample Data
    data = {
        'claim_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
        'proc_concat': [
            "0JB,0HZ",
            "0JB,0HT,09X",
            "09X,0HX",
            "0JZ,0JV",
            "0JB,0JZ",
            "0HX,0HT",
            "0JB",
            "09X,0JV",
            "0HZ,0HT,0JB",
            "0HX,0JZ,09X"
        ]
    }
    df_sample_claims = pd.DataFrame(data)

    # Generate AE features with a bottleneck of 16 dimensions
    ae_features_df = generate_ae_features(df_sample_claims, ae_bottleneck_dim=16)

    if ae_features_df is not None:
        print("\nResulting AE Features DataFrame (first 5 rows):")
        print(ae_features_df.head())
        print(f"\nShape of the resulting DataFrame: {ae_features_df.shape}")

    # Example with a claim having no codes (empty string or NaN)
    data_with_empty = {
        'claim_id': [2001, 2002, 2003],
        'proc_concat': [
            "0JB,0HZ",
            "", # Empty string
            np.nan # NaN
        ]
    }
    df_empty_claims = pd.DataFrame(data_with_empty)
    print("\n--- Testing with empty/NaN proc_concat ---")
    ae_features_empty_df = generate_ae_features(df_empty_claims, ae_bottleneck_dim=8)
    if ae_features_empty_df is not None:
        print("\nResulting AE Features DataFrame (with empty/NaN):")
        print(ae_features_empty_df)

    # Example with no unique codes overall
    data_no_codes = {
        'claim_id': [3001, 3002],
        'proc_concat': ["", np.nan]
    }
    df_no_codes = pd.DataFrame(data_no_codes)
    print("\n--- Testing with no unique codes overall ---")
    ae_features_no_codes_df = generate_ae_features(df_no_codes, ae_bottleneck_dim=4)

