from drgpy.msdrg import DRGEngine

def calculate_overpayment_risk(diag_list, proc_list, billed_drg, drg_weights, version="v40"):
    """
    Calculate the overpayment risk score for a given claim by permuting PDXs.

    Args:
        diag_list (list): Diagnosis codes on the claim.
        proc_list (list): Procedure codes on the claim.
        billed_drg (str): Reported DRG code from the claim.
        drg_weights (dict): Dict mapping DRG code to payment weight.
        version (str): MS-DRG grouper version (default 'v40').

    Returns:
        float: Overpayment risk score.
        list of dict: Detailed permutation results.
    """
    if billed_drg not in drg_weights:
        raise ValueError(f"Billed DRG code {billed_drg} missing from drg_weights mapping.")

    billed_weight = drg_weights[billed_drg]
    engine = DRGEngine(version=version)

    lower_weight_count = 0
    results = []

    for i, new_pdx in enumerate(diag_list):
        new_diag_list = [new_pdx] + diag_list[:i] + diag_list[i+1:]
        drg = engine.get_drg(new_diag_list, proc_list)
        weight = drg_weights.get(drg)
        # None weights are skipped from scoring
        if weight is not None and weight < billed_weight:
            lower_weight_count += 1
        results.append({
            "pdx": new_pdx,
            "drg": drg,
            "weight": weight,
        })

    risk_score = lower_weight_count / len(diag_list) if diag_list else 0
    return risk_score, results

if __name__ == "__main__":
    # ---- USER INPUT SECTION ----
    diag_list = ["E11.9", "I10", "N18.4"]
    proc_list = ["0DJ07ZZ"]
    billed_drg = "291"
    drg_weights = {
        "291": 1.0,
        "290": 0.8,
        "292": 1.2,
        # Add relevant DRGs here
    }
    # ---- END OF USER INPUT ----

    score, details = calculate_overpayment_risk(diag_list, proc_list, billed_drg, drg_weights, version="v40")
    
    print(f"Overpayment Risk Score: {score:.2f}")
    print("Permutation Results:")
    for d in details:
        print(f"PDX: {d['pdx']:<7} | DRG: {d['drg']:<5} | Weight: {d['weight']}")
