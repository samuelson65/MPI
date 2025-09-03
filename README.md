import re
from collections import defaultdict
from typing import Dict, List

class QueryMerger:
    def __init__(self):
        # Add common SQL operators
        self.operator_pattern = re.compile(r'\b(AND|OR)\b', re.IGNORECASE)
        self.condition_pattern = re.compile(
            r'([A-Z0-9_]+)\s*(=|!=|<>|>|<|>=|<=|IN|NOT IN|BETWEEN|NOT BETWEEN|LIKE|NOT LIKE|IS NULL|IS NOT NULL)\s*(.*)',
            re.IGNORECASE
        )

    def normalize(self, cond: str) -> str:
        cond = cond.upper().strip()
        cond = re.sub(r'\s+', ' ', cond)
        return cond

    def tokenize_conditions(self, condition: str) -> List[Dict]:
        """Tokenize query condition string into structured tokens"""
        tokens = []
        parts = self.operator_pattern.split(condition)
        current_op = "AND"
        for part in parts:
            part = part.strip()
            if part in ["AND", "OR"]:
                current_op = part
                continue
            match = self.condition_pattern.match(part)
            if match:
                field, op, value = match.groups()
                field, op, value = field.strip(), op.strip(), value.strip()
                values = self.extract_values(op, value)
                tokens.append({
                    "field": field,
                    "operator": op,
                    "values": values,
                    "logical_op": current_op
                })
        return tokens

    def extract_values(self, operator: str, value: str) -> List[str]:
        """Extract values based on operator type"""
        if operator in ["=", "!=", "<>", "LIKE", "NOT LIKE"]:
            return [value.strip("'\"")]
        elif operator in ["IN", "NOT IN"]:
            return [v.strip("'\" ") for v in re.split(r',', value.strip("()")) if v.strip()]
        elif operator in ["BETWEEN", "NOT BETWEEN"]:
            bounds = re.findall(r'\d+', value)
            return bounds if bounds else [value]
        elif operator in ["IS NULL", "IS NOT NULL"]:
            return []
        else:
            return [value]

    def merge_tokens(self, tokens_a: List[Dict], tokens_b: List[Dict]) -> List[Dict]:
        """Combine tokens intelligently with support for multiple operators"""
        combined = defaultdict(lambda: {"operator": None, "values": set(), "logical_op": "AND"})

        for token in tokens_a + tokens_b:
            field = token["field"]
            op = token["operator"]
            vals = token["values"]

            if combined[field]["operator"] in [None, "=", "IN"] and op in ["=", "IN"]:
                combined[field]["operator"] = "IN"
                combined[field]["values"].update(vals)
            elif combined[field]["operator"] in ["!=", "<>"] and op in ["!=", "<>"]:
                combined[field]["values"].update(vals)
                combined[field]["operator"] = "NOT IN"
            else:
                combined[field]["operator"] = op
                combined[field]["values"].update(vals)

            combined[field]["logical_op"] = token["logical_op"]

        merged_list = []
        for field, details in combined.items():
            merged_list.append({
                "field": field,
                "operator": details["operator"],
                "values": sorted(details["values"]),
                "logical_op": details["logical_op"]
            })
        return merged_list

    def build_condition(self, merged_tokens: List[Dict]) -> str:
        """Reconstructs the condition string from tokens"""
        parts = []
        for token in merged_tokens:
            field, op, values, logical_op = token["field"], token["operator"], token["values"], token["logical_op"]
            if op in ["IN", "NOT IN"]:
                parts.append(f"{field} {op} ({', '.join(values)})")
            elif op in ["IS NULL", "IS NOT NULL"]:
                parts.append(f"{field} {op}")
            elif op in ["BETWEEN", "NOT BETWEEN"] and len(values) == 2:
                parts.append(f"{field} {op} {values[0]} AND {values[1]}")
            else:
                parts.append(f"{field} {op} {values[0]}" if values else f"{field} {op}")
        return f" {logical_op} ".join(parts)

    def merge_queries(self, name_a: str, cond_a: str, name_b: str, cond_b: str) -> Dict:
        cond_a, cond_b = self.normalize(cond_a), self.normalize(cond_b)
        tokens_a = self.tokenize_conditions(cond_a)
        tokens_b = self.tokenize_conditions(cond_b)
        merged_tokens = self.merge_tokens(tokens_a, tokens_b)
        merged_condition = self.build_condition(merged_tokens)
        return {
            "merged_name": f"{name_a}_{name_b}_MERGED",
            "merged_condition": merged_condition
        }


# ----------------- USAGE EXAMPLES -----------------
merger = QueryMerger()

# 1️⃣ Simple equals and greater-than merge
print(merger.merge_queries(
    "QueryA", "DRG=871 AND LOS>10",
    "QueryB", "DRG=872 AND LOS>10"
))

# 2️⃣ Merge with NOT EQUAL and IN
print(merger.merge_queries(
    "QueryC", "DRG!=871 AND STATUS=DISCHARGED",
    "QueryD", "DRG!=872 AND STATUS=DISCHARGED"
))

# 3️⃣ Merge with LIKE and BETWEEN
print(merger.merge_queries(
    "QueryE", "PROC LIKE 'CARD%' AND LOS BETWEEN 5 AND 10",
    "QueryF", "PROC LIKE 'CARD%' AND LOS BETWEEN 7 AND 12"
))

# 4️⃣ Merge with IS NULL / IS NOT NULL
print(merger.merge_queries(
    "QueryG", "PROC IS NULL",
    "QueryH", "PROC IS NOT NULL"
))
