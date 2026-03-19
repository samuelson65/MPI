import pandas as pd

def clean_value(val):
    if val is None:
        return None
    return val.replace("'", "").replace("(", "").replace(")", "").strip()


def extract_values_from_in(line):
    try:
        start = line.upper().index("IN")
        vals = line[start:].split("(", 1)[1].rsplit(")", 1)[0]
        return [clean_value(x) for x in vals.split(",") if x.strip()]
    except:
        return []


def parse_sql(sql_text):
    results = []

    # Split statements safely
    statements = sql_text.split(";")

    for stmt in statements:
        if "SELECT" not in stmt.upper():
            continue

        lines = stmt.split("\n")

        selection_reason = None
        hcpcs_list = []
        paid_op = paid_val = None
        unit_op = unit_val = None
        mod_op = mod_val = None

        for line in lines:
            l = line.strip().upper()

            # SelectionReason
            if "SELECT DISTINCT" in l:
                try:
                    selection_reason = line.split("'")[1]
                except:
                    selection_reason = None

            # HCPCS
            if "HCPCSCODE IN" in l:
                hcpcs_list = extract_values_from_in(line)

            elif "HCPCSCODE =" in l:
                try:
                    hcpcs_list = [clean_value(line.split("=")[1])]
                except:
                    pass

            # PaidAmount
            if "PAIDAMOUNT" in l:
                for op in [">=", "<=", "<>", ">", "<", "="]:
                    if op in line:
                        try:
                            paid_op = op
                            paid_val = clean_value(line.split(op)[1])
                        except:
                            pass

            # Unit
            if "UNITCOUNT" in l:
                for op in [">=", "<=", "<>", ">", "<", "="]:
                    if op in line:
                        try:
                            unit_op = op
                            unit_val = clean_value(line.split(op)[1])
                        except:
                            pass

            # Modifier
            if "MODIFIER" in l:
                for op in ["IN", "<>", "=",]:
                    if op in line:
                        try:
                            mod_op = op
                            if op == "IN":
                                vals = extract_values_from_in(line)
                                mod_val = ",".join(vals)
                            else:
                                mod_val = clean_value(line.split(op)[1])
                        except:
                            pass

        # ✅ KEY: Never fail even if HCPCS missing
        if not hcpcs_list:
            hcpcs_list = [None]

        for code in hcpcs_list:
            results.append({
                "SelectionReason": selection_reason,
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

    df = pd.DataFrame(results)

    # Ensure exact column order
    columns = [
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

    for col in columns:
        if col not in df:
            df[col] = None

    return df[columns]


# -------- RUN --------
with open("input.sql", "r", encoding="utf-8") as f:
    sql_text = f.read()

df = parse_sql(sql_text)
df.to_excel("output.xlsx", index=False)

print("✅ DONE — output.xlsx generated successfully")
