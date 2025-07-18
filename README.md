import pandas as pd
from drgpy.msdrg import DRGEngine

# ----------- USER INPUT SECTION -----------
# Example: Fill this with your full DRG weight mapping per your version
DRG_WEIGHTS = {
    "291": 1.0,
    "290": 0.8,
    "292": 1.2,
    "193": 0.9,
    "194": 0.95,
    "195": 0.7,
    # Add all needed DRGs...
}

# Example DataFrame: Each row must have 'diag_list', 'proc_list', 'billed_drg'
df = pd.DataFrame({
    "claim_id": [1001, 1002],
    "diag_list": [["E11.9", "I10", "N18.4"], ["I63.9", "I10", "J44.9"]],
    "proc_list": [["0DJ07ZZ"], []],
    "billed_drg": ["291", "193"]
})
# ----------- END USER INPUT SECTION -------

DRG_ENGINE = DRGEngine(version="v40")

def overpayment_risk(row):
    diag_list = row['diag_list']
    proc_list = row['proc_list']
    billed_drg = row['billed_drg']

    billed_weight = DRG_WEIGHTS.get(billed_drg)
    if billed_weight is None or not diag_list:
        return pd.Series({"risk_score": None, "drg_permutations": None})

    lower_weight_count = 0
    permutations = []

    for i, pdx in enumerate(diag_list):
        diag_perm = [pdx] + diag_list[:i] + diag_list[i+1:]
        drg_code = DRG_ENGINE.get_drg(diag_perm, proc_list)
        weight = DRG_WEIGHTS.get(drg_code)

        permutations.append({"pdx": pdx, "drg": drg_code, "weight": weight})

        if weight is not None and weight < billed_weight:
            lower_weight_count += 1

    risk_score = round(lower_weight_count / len(diag_list), 2)
    return pd.Series({"risk_score": risk_score, "drg_permutations": permutations})

df[["risk_score", "drg_permutations"]] = df.apply(overpayment_risk, axis=1)

print(df[["claim_id", "billed_drg", "risk_score"]])
