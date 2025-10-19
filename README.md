import pandas as pd
import numpy as np

def categorize_drg_logic_auto(df,
                              hitrate_col='hitrate',
                              overpay_col='avg_overpayment',
                              volume_col='volume',
                              drg_col='drg_code',
                              high_percentile=0.70,
                              low_percentile=0.30):
    """
    Automatically classify DRGs into Include More / Exclude / Review
    based on percentile-based thresholds for each metric.

    Parameters:
        df : pd.DataFrame with ['drg_code', 'hitrate', 'avg_overpayment', 'volume']
        high_percentile : float (0-1), e.g. 0.70 means top 30% are "High"
        low_percentile  : float (0-1), e.g. 0.30 means bottom 30% are "Low"

    Returns:
        pd.DataFrame with classification columns:
            drg_code, hitrate, avg_overpayment, volume, cluster_label, category
    """

    df = df.copy()
    required = [drg_col, hitrate_col, overpay_col, volume_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Convert numerics
    for c in [hitrate_col, overpay_col, volume_col]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=[hitrate_col, overpay_col, volume_col])

    # --- Compute adaptive thresholds ---
    thresholds = {
        hitrate_col: {
            'low':  df[hitrate_col].quantile(low_percentile),
            'high': df[hitrate_col].quantile(high_percentile)
        },
        overpay_col: {
            'low':  df[overpay_col].quantile(low_percentile),
            'high': df[overpay_col].quantile(high_percentile)
        },
        volume_col: {
            'low':  df[volume_col].quantile(low_percentile),
            'high': df[volume_col].quantile(high_percentile)
        }
    }

    # --- Assign flags dynamically ---
    def flag_value(val, low, high):
        if val >= high:
            return "High"
        elif val <= low:
            return "Low"
        else:
            return "Medium"

    for metric in [hitrate_col, overpay_col, volume_col]:
        df[f"{metric}_flag"] = df[metric].apply(lambda x: flag_value(x,
                                                                     thresholds[metric]['low'],
                                                                     thresholds[metric]['high']))

    # --- Combine flags into category ---
    def decide(row):
        highs = [row[f"{hitrate_col}_flag"], row[f"{overpay_col}_flag"], row[f"{volume_col}_flag"]].count("High")
        lows = [row[f"{hitrate_col}_flag"], row[f"{overpay_col}_flag"], row[f"{volume_col}_flag"]].count("Low")
        if highs == 3:
            return "Include More"
        elif lows == 3:
            return "Exclude"
        else:
            return "Review"

    df['category'] = df.apply(decide, axis=1)

    # --- Human-readable label ---
    df['cluster_label'] = (
        df[f"{hitrate_col}_flag"] + " Hitrate & " +
        df[f"{overpay_col}_flag"] + " Overpayment & " +
        df[f"{volume_col}_flag"] + " Volume"
    )

    # --- Output tidy DataFrame ---
    out = df[[drg_col, hitrate_col, overpay_col, volume_col, 'cluster_label', 'category']].reset_index(drop=True)
    return out, thresholds


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    sample = pd.DataFrame({
        'drg_code': ['drg_100', 'drg_101', 'drg_102', 'drg_103', 'drg_104', 'drg_105', 'drg_106'],
        'hitrate': [0.70, 0.55, 0.25, 0.88, 0.20, 0.68, 0.33],
        'avg_overpayment': [5800, 3200, 1100, 7200, 800, 4500, 2000],
        'volume': [45, 120, 300, 25, 400, 95, 60]
    })

    result, used_thresholds = categorize_drg_logic_auto(sample)

    print("\n=== Adaptive DRG Classification ===")
    print(result.to_string(index=False))
    print("\n=== Thresholds Used (Auto) ===")
    print(pd.DataFrame(used_thresholds))
