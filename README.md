import re
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def clean_sql(sql):
    """Removes SQL comments and standardizes whitespace to prevent parsing errors."""
    sql = re.sub(r'--.*', '', sql) # Remove single line comments
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.S) # Remove block comments
    return " ".join(sql.split())

def parse_sql_to_dataframe(raw_sql):
    try:
        clean_text = clean_sql(raw_sql)
        
        # 1. Extract SelectionReason
        reason_match = re.search(r"SELECT\s+DISTINCT\s+'([^']+)'", clean_text, re.I)
        reason = reason_match.group(1) if reason_match else "Unknown"

        # 2. Extract ALL HCPCS codes
        # This regex looks for everything between 'HCPCCODE IN (' and the closing ')'
        hcpcs_block = re.search(r"HCPCCODE\s+IN\s*\((.*?)\)", clean_text, re.I)
        if not hcpcs_block:
            logging.warning(f"No HCPCS block found for {reason}")
            return pd.DataFrame()

        # Split by comma, strip spaces, and remove single quotes
        hcpcs_list = [c.strip().replace("'", "") for c in hcpcs_block.group(1).split(',')]
        # Filter out empty strings
        hcpcs_list = [c for c in hcpcs_list if c]

        # 3. Extract Operators and Values
        def get_val(field):
            # Handles field > 10, field>=10, field = 10
            m = re.search(rf"{field}\s*([>=<]+)\s*(\d+)", clean_text, re.I)
            return (m.group(1), m.group(2)) if m else ("", "")

        paid_op, paid_val = get_val("PAIDAMOUNT")
        unit_op, unit_val = get_val("UNITCOUNT")

        # 4. Create the flat structure
        data = [{
            "SelectionReason": reason,
            "HCPCS_operator": "=",
            "HCPCS": code,
            "Paidamount_operator": paid_op,
            "Paidamount": paid_val,
            "Modifier_operator": "", 
            "Modifier": "",
            "Unit_operator": unit_op,
            "Unit": unit_val,
            "Diag_operator": "",
            "Diag": ""
        } for code in hcpcs_list]
        
        return pd.DataFrame(data)

    except Exception as e:
        logging.error(f"Error parsing SQL: {e}")
        return pd.DataFrame()

# --- Test with your exact image data ---
sql_input = """
WHERE C1.HCPCCODE IN ('B4149', 'B4150', 'B4152', 'B4153', 'B4154', 'B4155', 'B4164')
and C1.UNITCOUNT > 15
AND C1.PAIDAMOUNT >= 500
"""

df = parse_sql_to_dataframe(sql_input)
print(f"Total codes captured: {len(df)}")
print(df.head(10))
