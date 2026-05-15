import pandas as pd

INPUT_FILE = "input.csv"
OUTPUT_FILE = "cleaned_output.csv"
BAD_ROWS_FILE = "bad_rows.csv"

def clean_numeric_column(series):
    """
    Cleans and converts a pandas Series to numeric safely.
    """
    # Convert to string for uniform processing
    series = series.astype(str).str.strip()

    # Remove common noise
    series = (
        series
        .str.replace(',', '', regex=False)   # remove हजार separators
        .str.replace(r'[^\d\.\-]', '', regex=True)  # keep digits, dot, minus
    )

    # Convert to numeric
    numeric_series = pd.to_numeric(series, errors='coerce')

    return numeric_series


def process_csv(file_path):
    chunksize = 100000  # adjust based on memory
    cleaned_chunks = []
    bad_rows_list = []

    for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
        chunk_original = chunk.copy()

        for col in chunk.columns:
            # Try converting column to numeric
            converted = clean_numeric_column(chunk[col])

            # Heuristic: if most values become numeric, treat column as numeric
            valid_ratio = converted.notna().mean()

            if valid_ratio > 0.8:  # threshold (tune if needed)
                # Capture bad rows BEFORE replacing
                bad_mask = converted.isna() & chunk[col].notna()
                if bad_mask.any():
                    bad_rows = chunk_original.loc[bad_mask, [col]]
                    bad_rows["column_name"] = col
                    bad_rows_list.append(bad_rows)

                chunk[col] = converted  # replace with cleaned numeric

        cleaned_chunks.append(chunk)

    # Combine all chunks
    final_df = pd.concat(cleaned_chunks, ignore_index=True)

    # Save cleaned data
    final_df.to_csv(OUTPUT_FILE, index=False)

    # Save bad rows if any
    if bad_rows_list:
        bad_df = pd.concat(bad_rows_list, ignore_index=True)
        bad_df.to_csv(BAD_ROWS_FILE, index=False)

    print("✅ Cleaning completed")
    print(f"Cleaned file saved to: {OUTPUT_FILE}")
    if bad_rows_list:
        print(f"⚠️ Bad rows saved to: {BAD_ROWS_FILE}")


if __name__ == "__main__":
    process_csv(INPUT_FILE)
