Act as a Data Engineer. I have a SQL query containing multiple selection rules (like 'EnteralNutrition', 'ParenteralNutrition'). I need you to parse the WHERE clause of this query and transform it into a flat Python list of dictionaries that matches an Excel schema.
​Rules for Transformation:
​Explode HCPCS: Create a new row for every single code found inside the HCPCCODE IN (...) clause. Do not skip any codes.
​Capture Operators: Extract both the operator (e.g., >=, >, =) and the value for PAIDAMOUNT and UNITCOUNT.
​Handle Modifiers: If the SQL contains MODIFIER <> 'KF' or similar, put <> in the Modifier_operator column and 'KF' in the Modifier column.
​Output Columns: The final result must have these exact keys: SelectionReason, HCPCS_operator, HCPCS, Paidamount_operator, Paidamount, Modifier_operator, Modifier, Unit_operator, Unit, Diag_operator, and Diag.
​Format the final output as a Pandas DataFrame so I can verify the count of HCPCS codes captured.
