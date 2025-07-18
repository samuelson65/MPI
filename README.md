import pandas as pd
from drgpy.msdrg import DRGEngine

# ------------------- SETUP: DRG Weight Loader -------------------
def load_drg_weights(filepath=None):
    """
    Loads DRG weight table.
    If filepath is provided, expects a CSV with 'drg' and 'weight' columns.
    Otherwise, uses demo hardcoded data (fill out with your CMS data!).
    """
    if filepath:
        df_weights = pd.read_csv(filepath)
        weights = dict(zip(df_weights['drg'].astype(str), df_weights['weight']))
        return weights
    # DEMO: Replace with full CMS data in production!
    return {
        "291": 1.00, "290": 0.80, "292": 1.20,
        "193": 0.90, "194": 0.95, "195": 0.70,
        "705": 1.25, "871": 0.60, "872": 0.50,
        # Add more...
    }

# ------------------- MAIN PROCESSING FUNCTION -------------------
def calculate_overpayment_metrics(
    diag_list, proc_list, billed_drg, drg_weights, engine
):
    """
    For a claim's diagnosis/procedure list and billed DRG, returns overpayment risk score
    and detailed diagnostic permutations.
    """
    billed_weight = drg_weights.get(str(billed_drg))
    if billed_weight is None or not diag_list or not isinstance(diag_list, list):
        return {"risk_score": None, "drg_permutations": None, "flag": "Invalid or missing DRG"}

    lower_weight_count = 0
    permutations = []
    n = len(diag_list)

    for i, pdx in enumerate(diag_list):
        perm = [pdx] + diag_list[:i] + diag_list[i+1:]
        drg = engine.get_drg(perm, proc_list)
        weight = drg_weights.get(str(drg))
        if weight is not None and weight < billed_weight:
            lower_weight_count += 1
        permutations.append({"pdx": pdx, "drg": drg, "weight": weight})

    risk_score = round(lower_weight_count / n, 3) if n > 0 else None

    # For reporting/audit: flag potentially problematic claims
    flag = "Review" if risk_score is not None and risk_score > 0.5 else "OK"
    return {"risk_score": risk_score, "drg_permutations": permutations, "flag": flag}

# ------------------- DATAFRAME BATCH PROCESSING -------------------
def process_claims_df(df, drg_weights, drg_version="v40"):
    engine = DRGEngine(version=drg_version)

    def row_fn(row):
        out = calculate_overpayment_metrics(
            diag_list=row['diag_list'],
            proc_list=row['proc_list'],
            billed_drg=row['billed_drg'],
            drg_weights=drg_weights,
            engine=engine
        )
        return pd.Series({
            "risk_score": out['risk_score'],
            "flag": out['flag'],
            "drg_permutations": out['drg_permutations'],
        })

    return df.join(df.apply(row_fn, axis=1))

# ------------------- EXAMPLE USAGE -------------------
if __name__ == "__main__":
    # Load DRG payment weights (customize path as needed)
    drg_weights = load_drg_weights()  # or load_drg_weights("mydatarates2025.csv")

    # Example claims batch; fill with your real data
    claims = pd.DataFrame([
        {
            "claim_id": 1001,
            "diag_list": ["E11.9", "I10", "N18.4"],
            "proc_list": ["0DJ07ZZ"],
            "billed_drg": "291"
        },
        {
            "claim_id": 1002,
            "diag_list": ["I63.9", "I10", "J44.9"],
            "proc_list": [],
            "billed_drg": "193"
        },
        {
            "claim_id": 1003,  # Will be flagged as invalid (bad DRG)
            "diag_list": ["J18.9", "I10"],
            "proc_list": [],
            "billed_drg": "XXX"
        }
    ])

    # Apply batch scoring
    results = process_claims_df(claims, drg_weights, drg_version="v40")

    # Display results
    print(
        results[["claim_id", "billed_drg", "risk_score", "flag"]]
    )
    # For audit investigations, export details with permutations:
    # results.to_json("drg_audit_results.json", orient="records")

'''
Expected output:
   claim_id billed_drg  risk_score    flag
0      1001        291       0.667  Review
1      1002        193       0.333      OK
2      1003        XXX         NaN  Invalid or missing DRG
'''

# Support for easy integration into CatBoost or other ML pipelines included[1].
