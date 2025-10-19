import pandas as pd
import numpy as np

def categorize_drg_logic(df,
                         hitrate_col='hitrate',
                         overpay_col='avg_overpayment',
                         volume_col='volume',
                         drg_col='drg_code',
                         hitrate_thresh=0.6,
                         overpay_thresh=3000,
                         volume_thresh=100):
    """
    Rule-based classification of DRGs into performance categories.

    Parameters:
        df : pd.DataFrame with ['drg_code', 'hitrate', 'avg_overpayment', 'volume']
        *_thresh : thresholds for defining high/low for each metric

    Returns:
        pd.DataFrame with ['drg_code', 'hitrate', 'avg_overpayment', 'volume', 'category']
    """
    df = df.copy()
    required = [drg_col, hitrate_col, overpay_col, volume_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # convert to numeric safely
    for c in [hitrate_col, overpay_col, volume_col]:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # assign flags
    df['hitrate_flag'] = np.where(df[hitrate_col] >= hitrate_thresh, 'High', 'Low')
    df['overpay_flag'] = np.where(df[overpay_col] >= overpay_thresh, 'High', 'Low')
    df['volume_flag']  = np.where(df[volume_col]  >= volume_thresh, 'High', 'Low')

    # build category logic
    def decide(row):
        highs = [row['hitrate_flag'], row['overpay_flag'], row['volume_flag']].count('High')
        if highs == 3:
            return "Include More"
        elif highs <= 1:
            return "Exclude"
        else:
            return "Review"

    df['category'] = df.apply(decide, axis=1)

    # create readable cluster label
    df['cluster_label'] = (
        df['hitrate_flag'] + " Hitrate & " +
        df['overpay_flag'] + " Overpayment & " +
        df['volume_flag'] + " Volume"
    )

    out = df[[drg_col, hitrate_col, overpay_col, volume_col, 'cluster_label', 'category']].reset_index(drop=True)
    return out


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    sample = pd.DataFrame({
        'drg_code': ['drg_100', 'drg_101', 'drg_102', 'drg_103', 'drg_104', 'drg_105', 'drg_106'],
        'hitrate': [0.70, 0.55, 0.25, 0.88, 0.20, 0.68, 0.33],
        'avg_overpayment': [5800, 3200, 1100, 7200, 800, 4500, 2000],
        'volume': [45, 120, 300, 25, 400, 95, 60]
    })

    result = categorize_drg_logic(sample)
    print(result.to_string(index=False))
