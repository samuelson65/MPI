import re
import pandas as pd

def safe_extract(pattern, text, group=1):
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return match.group(group)
        except IndexError:
            return None
    return None


def extract_hcpcs(block):
    # Try IN clause
    in_match = re.search(r"HCPCSCODE\s+IN\s*\((.*?)\)", block, re.IGNORECASE)
    if in_match:
        return [x.strip().replace("'", "") for x in in_match.group(1).split(",") if x.strip()]

    # Try equals
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


def parse_sql_file(sql_text):
    blocks = re.split(r"INSERT\s+INTO", sql_text, flags=re.IGNORECASE)

    all_rows = []

    for i, block in enumerate(blocks):
        if "SELECT" not in block:
            continue

        try:
            selection_reason = safe_extract(r"SELECT DISTINCT\s+'(.*?)'", block)
            hcpcs_list = extract_hcpcs(block)

            paid_op, paid_val = extract_condition(block, "PAIDAMOUNT")
            unit_op, unit_val = extract_condition(block, "UNITCOUNT")
            mod_op, mod_val = extract_modifier(block)

            # ⚠️ Skip blocks with no HCPCS (optional decision)
            if not hcpcs_list:
                print(f"⚠️ Skipping block {i} — No HCPCS found")
                continue

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

        except Exception as e:
            print(f"❌ Error in block {i}: {e}")
            continue

    if not all_rows:
        print("⚠️ No valid rows extracted!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Ensure all columns exist
    expected_cols = [
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

    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    df = df[expected_cols]

    return df


# ---------------- RUN ----------------

with open("input.sql", "r") as f:
    sql_text = f.read()

df = parse_sql_file(sql_text)

df.to_excel("output.xlsx", index=False)

print("✅ Done! Check output.xlsx")
