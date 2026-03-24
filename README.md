import pandas as pd
import re

# Your SQL snippet as a raw string
sql_content = """
--$CaptureLog[{"AuditType": "Exclusion", "AuditDescription": "HumanaEmployeeIndicator", "CaptureTotalTime": "True"}];
INSERT INTO ${temp.schema}.DME_SSPExclusionresults
select DISTINCT claimsummaryedenuid ,
'HumanaEmployeeIndicator',
ResultUID,
HumanaEmployeeIndicator_cs,
BillType,
ProviderTaxId,
SourcePlaceOfService,
CustomerID,
LoadMonth
from ${temp.schema}.KeepDMETarget_12mon
WHERE HumanaEmployeeIndicator_cs='Y';

--$CaptureLog[{"AuditType": "Exclusion", "AuditDescription": "ASO", "CaptureTotalTime": "True"}];
INSERT INTO ${temp.schema}.DME_SSPExclusionresults
select DISTINCT claimsummaryedenuid ,
'ASO',
ResultUID,
HumanaEmployeeIndicator_cs,
BillType,
ProviderTaxId,
SourcePlaceOfService,
CustomerID,
LoadMonth
from ${temp.schema}.KeepDMETarget_12mon
WHERE CustomerID in ('744207','TBD','747828','749282','760803','747744','760952','635506','761632');
"""

def parse_sql_to_table(sql_text):
    rules = []
    
    # Split by CaptureLog markers to separate blocks
    blocks = re.split(r'--\$CaptureLog', sql_text)
    
    for block in blocks:
        if not block.strip():
            continue
            
        # Extract the SelectionReason (AuditDescription)
        reason_match = re.search(r'"AuditDescription":\s*"([^"]+)"', block)
        reason = reason_match.group(1) if reason_match else "Unknown"
        
        # Extract the WHERE clause filter logic
        where_match = re.search(r'WHERE\s+(.*);', block, re.IGNORECASE | re.DOTALL)
        condition = where_match.group(1).strip() if where_match else ""
        
        # Determine operator and value based on the condition
        # This handles both '=' and 'IN' logic
        if 'in' in condition.lower():
            operator = "IN"
            # Extract values inside the parentheses
            values = re.search(r'\((.*)\)', condition).group(1)
            column = condition.split()[0]
        elif '=' in condition:
            operator = "="
            parts = condition.split('=')
            column = parts[0].strip()
            values = parts[1].strip().replace("'", "")
        else:
            operator = "N/A"
            column = "N/A"
            values = condition

        rules.append({
            "SelectionReason": reason,
            "Filter_Column": column,
            "Operator": operator,
            "Value": values
        })
        
    return pd.DataFrame(rules)

# Generate the DataFrame
df_rules = parse_sql_to_table(sql_content)

# Display the result
print("--- Structured Rule Table ---")
print(df_rules.to_string(index=False))

# Optional: Export to Excel to match your screenshot
# df_rules.to_excel("DME_Exclusion_Rules.xlsx", index=False)
