
import pandas as pd
import numpy as np
from drg_ruleminer import DRGRuleMiner

# Sample dummy claims dataset
np.random.seed(42)
df = pd.DataFrame({
    "mcc_count": np.random.randint(0, 6, 1000),
    "cc_count": np.random.randint(0, 4, 1000),
    "procedure_count": np.random.randint(0, 8, 1000),
    "length_of_stay": np.random.randint(1, 15, 1000),
    "discharge_status": np.random.choice([1, 2, 3, 4, 5], 1000),
    "overpayment": np.random.choice([0, 1], 1000, p=[0.8, 0.2])
})

X = df.drop(columns="overpayment")
y = df["overpayment"]

miner = DRGRuleMiner(max_depth=4, min_samples_leaf=20)
miner.fit(X, y)

print("\nTop Rules:")
print(miner.rules.head())

print("\nSQL Query for Top Rules:")
print(miner.to_sql(miner.export_rules()))        candidate_rules = self._tree_to_rules(self.tree.tree_, self.feature_names)
        self.rules = self._evaluate_rules(candidate_rules, X, y)

        return self

    def _evaluate_rules(self, rules: List[str], X: pd.DataFrame, y: pd.Series):
        results = []
        for rule in rules:
            try:
                mask = X.eval(rule)
            except Exception:
                continue
            if mask.sum() == 0:
                continue
            precision = precision_score(y[mask], np.ones(mask.sum()), zero_division=0)
            recall = recall_score(y, mask, zero_division=0)
            lift = precision / (y.mean() + 1e-9)
            results.append({
                "rule": rule,
                "support": mask.sum(),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "lift": round(lift, 4)
            })
        return pd.DataFrame(results).sort_values(by=["precision", "lift"], ascending=False)

    def export_rules(self, top_n: int = 10) -> List[str]:
        if self.rules is None or len(self.rules) == 0:
            return []
        return self.rules.head(top_n)["rule"].tolist()

    def to_sql(self, rules: List[str]) -> str:
        sql_rules = []
        for rule in rules:
            sql_rule = rule.replace("and", "AND").replace("or", "OR")
            sql_rules.append(f"({sql_rule})")
        return " OR ".join(sql_rules)
