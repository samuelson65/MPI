import re
from collections import defaultdict
from typing import Dict, List

class QueryMerger:
    def __init__(self):
        self.operator_pattern = re.compile(r'\b(AND|OR)\b', re.IGNORECASE)
        self.condition_pattern = re.compile(
            r'([A-Z0-9_]+)\s*(=|>|<|>=|<=|IN|BETWEEN)\s*(.*)', re.IGNORECASE
        )

    def normalize(self, cond: str) -> str:
        cond = cond.upper().strip()
        cond = re.sub(r'\s+', ' ', cond)
        return cond

    def tokenize_conditions(self, condition: str) -> List[Dict]:
        """Breaks condition into tokens: field, operator, values"""
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
        if operator == "=":
            return [value]
        elif operator == "IN":
            return [v.strip() for v in re.findall(r'\d+|\w+', value)]
        elif operator == "BETWEEN":
            bounds = re.findall(r'\d+', value)
            return [str(bounds[0]), str(bounds[1])] if len(bounds) == 2 else []
        else:
            return [value]

    def merge_tokens(self, tokens_a: List[Dict], tokens_b: List[Dict]) -> List[Dict]:
        """Combine tokens intelligently"""
        combined = defaultdict(lambda: {"operator": None, "values": set(), "logical_op": "AND"})

        for token in tokens_a + tokens_b:
            field = token["field"]
            op = token["operator"]
            if combined[field]["operator"] in [None, "=", "IN"] and op in ["=", "IN"]:
                combined[field]["operator"] = "IN"
                combined[field]["values"].update(token["values"])
            else:
                combined[field]["operator"] = op
                combined[field]["values"].update(token["values"])
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
        """Builds query string from merged tokens"""
        parts = []
        for token in merged_tokens:
            field, op, values, logical_op = token["field"], token["operator"], token["values"], token["logical_op"]
            if op == "IN":
                parts.append(f"{field} IN ({', '.join(values)})")
            elif op == "=":
                parts.append(f"{field}={values[0]}")
            elif op == "BETWEEN":
                parts.append(f"{field} BETWEEN {values[0]} AND {values[1]}")
            else:
                parts.append(f"{field}{op}{values[0]}")
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

# ------------------ EXAMPLES ------------------
merger = QueryMerger()

# Case 1: Simple merge with equality
q1 = "DRG=871 AND LOS>10"
q2 = "DRG=872 AND LOS>10"
print(merger.merge_queries("QueryA", q1, "QueryB", q2))

# Case 2: Merge with IN operator
q3 = "DRG IN (870, 871) AND LOS>5"
q4 = "DRG=872 AND LOS>5"
print(merger.merge_queries("QueryC", q3, "QueryD", q4))

# Case 3: Merge with BETWEEN and OR
q5 = "DRG=871 OR LOS BETWEEN 5 AND 10"
q6 = "DRG=872 OR LOS BETWEEN 7 AND 12"
print(merger.merge_queries("QueryE", q5, "QueryF", q6))
