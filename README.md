import re
import pandas as pd

# -------------------------------
# Helper Functions
# -------------------------------

def extract_selection_reason(block):
    match = re.search(r"SELECT DISTINCT\s+'(.*?)'", block, re.IGNORECASE)
    return match.group(1) if match else None


def extract_hcpcs(block):
    # IN clause
    in_match = re.search(r"HCPCSCODE\s+IN\s*\((.*?)\)", block, re.IGNORECASE)
    if in_match:
        return [x.strip().replace("'", "") for x in in_match.group(1).split(",")]

    # = clause
    eq_match = re.search(r"HCPCSCODE\s*=\s*'(.*?)'", block, re.IGNORECASE)
    if eq_match:
        return [eq_match.group(1)]

    return []


def extract_condition(block, field):
    match = re.search(fr"{field}\s*(>=|<=|<>|=|>|<)\s*'?([\w\.]+)'?", block, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)
    return None, None


def extract_modifier(block):
    match = re.search(r"MODIFIER\s*(=|<>|IN)\s*(.*?)(AND|OR|$)", block, re.IGNORECASE)
    if match:
        op = match.group(1)
        val = match.group(2).strip().replace("'", "")
        return op, val
    return None, None


# -------------------------------
# Main Parser
# -------------------------------

def parse_sql_file(sql_text):
    # Split into blocks (each INSERT/SELECT)
    blocks = re.split(r"INSERT\s+INTO", sql_text, flags=re.IGNORECASE)

    all_rows = []

    for block in blocks:
        if "SELECT" not in block:
            continue

        selection_reason = extract_selection_reason(block)
        hcpcs_list = extract_hcpcs(block)

        paid_op, paid_val = extract_condition(block, "PAIDAMOUNT")
        unit_op, unit_val = extract_condition(block, "UNITCOUNT")
        mod_op, mod_val = extract_modifier(block)

        # Expand rows
        for code in hcpcs_list:
            row = {
                "SelectionReason": selection_reason,
                "HCPCS_operator": "=",
                "HCPCS": code,
                "Paidamount_operator": paid_op,
                "Paidamount": paid_val,
                "Modifier_operator": mod_op,
                "Modifier": mod_val,
                "Unit_operator": unit_op,
                "Unit": unit_val,
                "Diag_operator": None,
                "Diag": None
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # Ensure column order
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


# -------------------------------
# Run Script
# -------------------------------

with open("input.sql", "r") as f:
    sql_text = f.read()

df = parse_sql_file(sql_text)

# Save Excel
df.to_excel("output.xlsx", index=False)

print("✅ Full SQL converted to structured Excel: output.xlsx")
