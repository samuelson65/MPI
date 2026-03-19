import re
import pandas as pd

def parse_sql_to_excel_format(sql):
    # -------- Selection Reason --------
    selection_match = re.search(r"SELECT DISTINCT\s+'(.*?)'", sql, re.IGNORECASE)
    selection_reason = selection_match.group(1) if selection_match else None

    # -------- HCPCS Codes --------
    hcpcs_match = re.search(r"HCPCSCODE\s+IN\s*\((.*?)\)", sql, re.IGNORECASE)
    hcpcs_codes = []
    if hcpcs_match:
        hcpcs_codes = [x.strip().replace("'", "") for x in hcpcs_match.group(1).split(",")]

    # -------- Paid Amount --------
    paid_match = re.search(r"PAIDAMOUNT\s*(>=|>|=|<=|<)\s*(\d+)", sql, re.IGNORECASE)
    paid_op, paid_val = (paid_match.group(1), paid_match.group(2)) if paid_match else (None, None)

    # -------- Unit --------
    unit_match = re.search(r"UNITCOUNT\s*(>=|>|=|<=|<)\s*(\d+)", sql, re.IGNORECASE)
    unit_op, unit_val = (unit_match.group(1), unit_match.group(2)) if unit_match else (None, None)

    # -------- Modifier (optional) --------
    modifier_match = re.search(r"MODIFIER\s*(=|<>|IN)\s*(.*)", sql, re.IGNORECASE)
    modifier_op, modifier_val = (None, None)
    if modifier_match:
        modifier_op = modifier_match.group(1)
        modifier_val = modifier_match.group(2).strip()

    # -------- Diagnosis (optional) --------
    diag_match = re.search(r"DIAG\s*(=|<>|IN)\s*(.*)", sql, re.IGNORECASE)
    diag_op, diag_val = (None, None)
    if diag_match:
        diag_op = diag_match.group(1)
        diag_val = diag_match.group(2).strip()

    # -------- Build Output --------
    rows = []
    for code in hcpcs_codes:
        rows.append({
            "SelectionReason": selection_reason,
            "HCPCS_operator": "=",
            "HCPCS": code,
            "Paidamount_operator": paid_op,
            "Paidamount": paid_val,
            "Modifier_operator": modifier_op,
            "Modifier": modifier_val,
            "Unit_operator": unit_op,
            "Unit": unit_val,
            "Diag_operator": diag_op,
            "Diag": diag_val
        })

    df = pd.DataFrame(rows)

    # Ensure column order matches EXACT Excel
    df = df[
        [
            "SelectionReason",
            "HCPCS_operator",
            "HCPCS",
            "Paidamount_operator",
            "Paidamount",
            "Modifier_operator",
            "Modifier",
            "Unit_operator",
            "Unit",
            "Diag_operator",
            "Diag"
        ]
    ]

    return df


# -------- Usage --------
with open("input.sql", "r") as f:
    sql_query = f.read()

df = parse_sql_to_excel_format(sql_query)

df.to_excel("output.xlsx", index=False)

print("✅ Output saved as output.xlsx")
