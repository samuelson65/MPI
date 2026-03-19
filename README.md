import re
import pandas as pd
import logging

# Setup logging to track errors without crashing the script
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def extract_sql_blocks(full_script):
    """Splits a large SQL file into individual INSERT/SELECT blocks."""
    # Matches starting from INSERT INTO up until the next block or end of string
    blocks = re.split(r'(?=INSERT INTO)', full_script)
    return [b.strip() for b in blocks if b.strip()]

def robust_sql_to_dataframe(sql_query):
    try:
        # 1. SelectionReason (Case-insensitive search)
        reason_match = re.search(r"SELECT\s+DISTINCT\s+'([^']+)'", sql_query, re.I)
        selection_reason = reason_match.group(1) if reason_match else "Unknown"

        # 2. HCPCS codes (Handles single quotes and spaces)
        hcpcs_match = re.search(r"HCPCCODE\s+IN\s*\((.*?)\)", sql_query, re.I | re.S)
        if not hcpcs_match:
            logging.warning(f"No HCPCS codes found for {selection_reason}")
            return pd.DataFrame()
        
        hcpcs_list = [c.strip().strip("'") for c in hcpcs_match.group(1).split(',')]

        # 3. Numeric Filters (Paid Amount & Unit Count)
        # Regex explanation: (Field Name) (Operator: >=, >, <, =, etc.) (Value: Digits)
        def get_filter(field_name):
            pattern = rf"{field_name}\s*([>=<]+)\s*(\d+)"
            match = re.search(pattern, sql_query, re.I)
            return (match.group(1), match.group(2)) if match else ("", "")

        paid_op, paid_val = get_filter("PAIDAMOUNT")
        unit_op, unit_val = get_filter("UNITCOUNT")

        # 4. Build Data
        rows = []
        for code in hcpcs_list:
            rows.append({
                "SelectionReason": selection_reason,
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
            })
        
        return pd.DataFrame(rows)

    except Exception as e:
        logging.error(f"Failed to parse block: {str(e)}")
        return pd.DataFrame()

def process_queries(raw_sql_text):
    """Main Orchestrator"""
    blocks = extract_sql_blocks(raw_sql_text)
    all_dfs = []
    
    for block in blocks:
        df_block = robust_sql_to_dataframe(block)
        if not df_block.empty:
            all_dfs.append(df_block)
    
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        # Ensure 'Paidamount' and 'Unit' are treated as numbers where possible
        final_df['Paidamount'] = pd.to_numeric(final_df['Paidamount'], errors='coerce')
        final_df['Unit'] = pd.to_numeric(final_df['Unit'], errors='coerce')
        return final_df
    
    return pd.DataFrame()

# --- Execution ---
# You can paste the entire content of your SQL file here
raw_input = """
-- EnteralNutrition
INSERT INTO ${temp.schema}.dme_ssp_final
SELECT DISTINCT 'EnteralNutrition' AS SELECTIONREASON, ...
WHERE C1.HCPCCODE IN ('B4149', 'B4150', 'B4152')
AND C1.UNITCOUNT > 15
AND C1.PAIDAMOUNT >= 500;

-- ParenteralNutrition
INSERT INTO ${temp.schema}.dme_ssp_final
SELECT DISTINCT 'ParenteralNutrition' AS SELECTIONREASON, ...
WHERE C1.HCPCCODE IN ('B4185', 'B4189')
AND C1.UNITCOUNT > 20
AND C1.PAIDAMOUNT >= 200;
"""

final_output = process_queries(raw_input)

# Save to Excel
if not final_output.empty:
    final_output.to_excel("Selection_Criteria_Output.xlsx", index=False)
    print("Success! File saved as Selection_Criteria_Output.xlsx")
else:
    print("No data was parsed. Check your SQL input format.")
