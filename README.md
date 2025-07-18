import pandas as pd
from drgpy.msdrg import DRGEngine

# DRG engine setup (set once and reuse)
drg_engine = DRGEngine(version="v40")

def compute_overpayment_score(diag_list, proc_list, billed_drg, drg_weights):
    """
    Calculates overpayment risk score by permuting all PDX positions.
    """
    if not isinstance(diag_list, list) or not diag_list:
        return None, None  # No diagnosis codes

    billed_weight = drg_weights.get(str(billed_drg))
    if billed_weight is None:
        return None, None  # DRG not in weight table

    lower_weight_count = 0
    total_diag = len(diag_list)
    permutations = []

    for i, pdx in enumerate(diag_list):
        permuted_diags = [pdx] + diag_list[:i] + diag_list[i+1:]
        derived_drg = drg_engine.get_drg(permuted_diags, proc_list or [])
        derived_weight = drg_weights.get(str(derived_drg))

        if derived_weight is not None and derived_weight < billed_weight:
            lower_weight_count += 1

        permutations.append({
            "pdx": pdx,
            "derived_drg": derived_drg,
            "derived_weight": derived_weight
        })

    score = round(lower_weight_count / total_diag, 3)
    return score, permutations

def add_overpayment_scores(df, drg_weights):
    """
    Adds 'risk_score' and 'drg_permutations' columns to a claim DataFrame.
    """

    def row_score(row):
        score, perms = compute_overpayment_score(
            diag_list=row['diag_list'],
            proc_list=row.get('proc_list', []),
            billed_drg=row['drgcode'],
            drg_weights=drg_weights
        )
        return pd.Series({
            "risk_score": score,
            "drg_permutations": perms
        })

    return df.join(df.apply(row_score, axis=1))

# ------------- Example Usage -------------

if __name__ == "__main__":
    # ✅ Your DRG weights dictionary
    drg_weights = {
        "291": 1.0,
        "290": 0.8,
        "292": 1.2,
        "193": 0.9,
        "194": 0.95,
        "195": 0.7,
        # Add all needed DRGs
    }

    # ✅ Your DataFrame: Each row represents a claim
    df = pd.DataFrame({
        "claim_id": [1001, 1002],
        "diag_list": [["E11.9", "I10", "N18.4"], ["I63.9", "J44.9", "I10"]],
        "proc_list": [["0DJ07ZZ"], []],
        "drgcode": ["291", "193"]
    })

    df = add_overpayment_scores(df, drg_weights)

    # ✅ Output result
    print(df[["claim_id", "drgcode", "risk_score"]])
    # Optionally view permutations too:
    # print(df[["claim_id", "drg_permutations"]])
