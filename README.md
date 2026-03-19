import re
import pandas as pd


def extract_first(pattern, text):
    """Return first regex group safely"""
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else None


def extract_operator_value(pattern, text):
    """Extract operator + value safely"""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)
    return None, None


def extract_hcpcs(block):
    # IN clause
    match = re.search(r"HCPCSCODE\s+IN\s*\((.*?)\)", block, re.IGNORECASE | re.DOTALL)
    if match:
        values = match.group(1)
        return [x.strip().replace("'", "") for x in values.split(",") if x.strip()]

    # = clause
    match = re.search(r"HCPCSCODE\s*=\s*'(.*?)'", block, re.IGNORECASE)
    if match:
        return [match.group(1)]

    return []


def extract_modifier(block):
    # Handles =, <>, IN
    match = re.search(r"MODIFIER\s*(=|<>|IN)\s*\(?([^)]+)\)?", block, re.IGNORECASE)
    if match:
        op = match.group(1)
        val = match.group(2).replace("'", "").strip()
        return op, val
    return None, None


def parse_sql(sql_text):
    # Split by INSERT blocks (better for your structure)
    blocks = re.split(r"INSERT\s+INTO", sql_text, flags=re.IGNORECASE)

    rows = []

    for block in blocks:
        if "SELECT" not in block.upper():
            continue

        # -------- Core fields --------
        selection = extract_first(r"SELECT\s+DISTINCT\s+'(.*?)'", block)
        hcpcs_list = extract_hcpcs(block)

        paid_op, paid_val = extract_operator_value(
            r"PAIDAMOUNT\s*(>=|<=|<>|=|>|<)\s*(\d+)", block
        )

        unit_op, unit_val = extract_operator_value(
            r"UNITCOUNT\s*(>=|<=|<>|=|>|<)\s*(\d+)", block
        )

        mod_op, mod_val = extract_modifier(block)

        # ✅ FIX: don't crash if HCPCS missing
        if not hcpcs_list:
            hcpcs_list = [None]

        for code in hcpcs_list:
            rows.append({
                "SelectionReason": selection,
                "HCPCS_operator": "=" if code else None,
                "HCPCS": code,
                "Paidamount_operator": paid_op,
                "Paidamount": paid_val,
                "Modifier_operator": mod_op,
                "Modifier": mod_val,
                "Unit_operator": unit_op,
                "Unit": unit_val,
                "Diag_operator": None,
                "Diag": None
            })

    df = pd.DataFrame(rows)

    # Ensure exact format
    cols = [
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

    for col in cols:
        if col not in df.columns:
            df[col] = None

    return df[cols]


# -------- RUN --------
with open("input.sql", "r", encoding="utf-8") as f:
    sql_text = f.read()

df = parse_sql(sql_text)
df.to_excel("output.xlsx", index=False)

print("✅ output.xlsx created")
