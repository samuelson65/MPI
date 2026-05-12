"""
CPT Code Denial Risk Analysis
------------------------------
Flattens cpt_findingline_dict, maps outcomes, aggregates by CPT code,
and ranks codes by drop percentage to surface denial risk.
"""

import pandas as pd
import ast


# ── 1. Load / define your data ────────────────────────────────────────────────
# Replace this with pd.read_csv(...) or however you load your real data.
sample_data = [
    {
        "cpt_findingline_dict": {"99213": 1, "93000": 0, "85025": 1},
        "findings_status": "Partial Takeback",
    },
    {
        "cpt_findingline_dict": {"99213": 0, "99214": 1, "93000": 1},
        "findings_status": "Full Takeback",
    },
    {
        "cpt_findingline_dict": {"85025": 0, "99213": 1, "99215": 0},
        "findings_status": "No Findings",
    },
    {
        "cpt_findingline_dict": {"99213": 1, "93000": 1},
        "findings_status": "Full Takeback",
    },
    {
        "cpt_findingline_dict": {"99214": 0, "85025": 1, "99215": 1},
        "findings_status": "Partial Takeback",
    },
]

df = pd.DataFrame(sample_data)

# If your dict column is stored as a string (e.g., from CSV), parse it first:
# df["cpt_findingline_dict"] = df["cpt_findingline_dict"].apply(ast.literal_eval)


# ── 2. Map findings_status → is_dropped ──────────────────────────────────────
takeback_statuses = {"Partial Takeback", "Full Takeback"}
df["is_dropped"] = df["findings_status"].isin(takeback_statuses).astype(int)


# ── 3. Flatten: one row per CPT occurrence ────────────────────────────────────
rows = []
for _, row in df.iterrows():
    is_dropped = row["is_dropped"]
    for cpt_code, denial_flag in row["cpt_findingline_dict"].items():
        rows.append(
            {
                "CPT_Code": cpt_code,
                "denial_flag": denial_flag,   # 1 = denied at line level
                "is_dropped": is_dropped,      # 1 = claim-level takeback
            }
        )

long_df = pd.DataFrame(rows)


# ── 4. Aggregate by CPT code ──────────────────────────────────────────────────
summary = (
    long_df.groupby("CPT_Code")
    .agg(
        Total_Occurrences=("is_dropped", "count"),
        Total_Drops=("is_dropped", "sum"),
    )
    .reset_index()
)

summary["Drop_Percentage"] = (
    summary["Total_Drops"] / summary["Total_Occurrences"] * 100
).round(2)


# ── 5. Filter low-volume CPTs & sort by risk ──────────────────────────────────
MIN_OCCURRENCES = 30  # adjust as needed

high_risk = (
    summary[summary["Total_Occurrences"] >= MIN_OCCURRENCES]
    .sort_values("Drop_Percentage", ascending=False)
    .reset_index(drop=True)
)

print("=== CPT Denial Risk Ranking ===")
print(high_risk.to_string(index=False))


# ── 6. Score a NEW claim ─────────────────────────────────────────────────────
def score_new_claim(new_cpt_dict: dict, risk_table: pd.DataFrame) -> pd.DataFrame:
    """
    Given a new claim's CPT dict and the pre-computed risk table,
    return a sorted breakdown of each code's denial risk.

    Parameters
    ----------
    new_cpt_dict : dict
        e.g. {"99213": 1, "85025": 0}
    risk_table : pd.DataFrame
        Output of the aggregation step above.

    Returns
    -------
    pd.DataFrame  sorted by Drop_Percentage descending
    """
    codes = list(new_cpt_dict.keys())
    scored = risk_table[risk_table["CPT_Code"].isin(codes)].copy()

    # Flag codes that are also denied at line level in this claim
    scored["Denied_In_Claim"] = scored["CPT_Code"].map(new_cpt_dict)

    # Mark codes with no historical data
    missing = set(codes) - set(scored["CPT_Code"])
    if missing:
        print(f"\n⚠️  No historical data for CPT(s): {missing}")

    return scored.sort_values("Drop_Percentage", ascending=False).reset_index(drop=True)


# Example new claim
new_claim = {"99213": 1, "85025": 0, "99999": 1}  # 99999 has no history
print("\n=== New Claim Denial Risk Breakdown ===")
print(score_new_claim(new_claim, high_risk).to_string(index=False))
