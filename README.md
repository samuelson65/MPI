import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def classify_drg_opportunity(df: pd.DataFrame,
                             feature_cols=None,
                             verbose=True,
                             plot=True):
    """
    Classify DRGs into High / Moderate / Low Opportunity groups based on
    volume, hitrate, and overpayment metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing DRG metrics.
        Must include: drgcode, hitrate, avg_overpayment, volume
    feature_cols : list, optional
        Defaults to ['median_probability', 'hitrate', 'avg_overpayment', 'volume']
    verbose : bool
        Whether to print summary insights.
    plot : bool
        Whether to show scatter plot visualization.

    Returns
    -------
    pd.DataFrame with columns:
        ['drgcode', 'category', 'score', 'volume', 'hitrate', 'avg_overpayment']
    """

    # --- Standardize column names ---
    df = df.rename(columns={c: c.lower() for c in df.columns})

    if feature_cols is None:
        feature_cols = ['median_probability', 'hitrate', 'avg_overpayment', 'volume']
    feature_cols = [c.lower() for c in feature_cols]

    required = ['drgcode'] + feature_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- Drop invalid rows ---
    df = df.dropna(subset=feature_cols).copy()
    if df.empty:
        raise ValueError("No valid rows after dropping NaNs.")

    # --- Compute quantiles for adaptive thresholds ---
    q = df[feature_cols].quantile([0.33, 0.66])
    low_q, high_q = q.iloc[0], q.iloc[1]

    # --- Normalize features and compute weighted opportunity score ---
    # You can adjust the weights as per business priority
    df['score'] = (
        0.4 * (df['hitrate'] - df['hitrate'].min()) / (df['hitrate'].max() - df['hitrate'].min() + 1e-9) +
        0.3 * (df['avg_overpayment'] - df['avg_overpayment'].min()) / (df['avg_overpayment'].max() - df['avg_overpayment'].min() + 1e-9) +
        0.3 * (df['volume'] - df['volume'].min()) / (df['volume'].max() - df['volume'].min() + 1e-9)
    )

    # --- Apply quantile-based classification ---
    conditions = [
        (df['score'] >= df['score'].quantile(0.66)),
        (df['score'].between(df['score'].quantile(0.33), df['score'].quantile(0.66))),
        (df['score'] < df['score'].quantile(0.33))
    ]
    categories = ['High-Opportunity', 'Moderate', 'Low-Opportunity']
    df['category'] = np.select(conditions, categories)

    # --- Optional overrides for strong rules ---
    df.loc[
        (df['hitrate'] > high_q['hitrate']) &
        (df['avg_overpayment'] > high_q['avg_overpayment']) &
        (df['volume'] > high_q['volume']),
        'category'
    ] = 'High-Opportunity'

    df.loc[
        (df['hitrate'] < low_q['hitrate']) &
        (df['avg_overpayment'] < low_q['avg_overpayment']) &
        (df['volume'] < low_q['volume']),
        'category'
    ] = 'Low-Opportunity'

    # --- Prepare final output ---
    result = df[['drgcode', 'category', 'score', 'volume', 'hitrate', 'avg_overpayment']].sort_values(
        'score', ascending=False
    )

    # --- Print summary ---
    if verbose:
        print("\n📊 DRG Opportunity Classification Summary:")
        dist = result['category'].value_counts(normalize=True).mul(100).round(1).to_dict()
        print("  Category distribution (%):", dist)
        print("\n🔝 Example High-Opportunity DRGs:")
        print(result[result['category'] == 'High-Opportunity'].head(5))

    # --- Visualization ---
    if plot:
        plt.figure(figsize=(8,6))
        colors = {
            'High-Opportunity': 'green',
            'Moderate': 'orange',
            'Low-Opportunity': 'red'
        }
        for cat, color in colors.items():
            subset = result[result['category'] == cat]
            plt.scatter(subset['hitrate'], subset['avg_overpayment'],
                        s=subset['volume'] * 0.5,  # size by volume
                        c=color, alpha=0.6, label=cat, edgecolor='k')

        plt.xlabel('Hitrate')
        plt.ylabel('Average Overpayment')
        plt.title('DRG Opportunity Classification')
        plt.legend(title='Category')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    return result


# ------------------- #
# Example usage below #
# ------------------- #
if __name__ == "__main__":
    # Example data
    data = {
        'drgcode': [291, 292, 293, 294, 295, 296, 297, 298],
        'median_probability': [0.81, 0.55, 0.91, 0.35, 0.62, 0.77, 0.45, 0.84],
        'hitrate': [0.73, 0.33, 0.89, 0.25, 0.6, 0.7, 0.4, 0.82],
        'avg_overpayment': [5600, 2500, 7000, 1100, 3500, 5200, 1800, 6400],
        'volume': [45, 150, 22, 420, 180, 55, 300, 40]
    }

    df = pd.DataFrame(data)

    result_df = classify_drg_opportunity(df, verbose=True, plot=True)

    print("\n✅ Final Output:")
    print(result_df)
