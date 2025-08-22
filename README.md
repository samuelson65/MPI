import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from typing import List

class DRGRuleMiner:
    def __init__(self, max_depth=4, min_samples_leaf=50, class_weight="balanced"):
        """
        DRG Rule Miner using decision tree extraction.
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.tree = None
        self.feature_names = None
        self.rules = pd.DataFrame()

    def _tree_to_rules(self, tree, feature_names):
        """
        Convert decision tree structure into human-readable rules.
        """
        rules = []

        def recurse(node, path):
            if tree.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_names[tree.feature[node]]
                threshold = tree.threshold[node]
                recurse(tree.children_left[node], path + [f"({name} <= {threshold:.4f})"])
                recurse(tree.children_right[node], path + [f"({name} > {threshold:.4f})"])
            else:
                if path:  # only record if path is not empty
                    rules.append(" AND ".join(path))

        recurse(0, [])
        return rules

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit decision tree and extract/evaluate rules.
        """
        self.feature_names = list(X.columns)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        # Train the decision tree
        self.tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=42
        )
        self.tree.fit(X_train, y_train)

        # Extract candidate rules
        candidate_rules = self._tree_to_rules(self.tree.tree_, self.feature_names)

        # Evaluate rules
        self.rules = self._evaluate_rules(candidate_rules, X, y)

        return self

    def _evaluate_rules(self, rules: List[str], X: pd.DataFrame, y: pd.Series):
        """
        Evaluate rules with precision, recall, and lift.
        """
        if not rules:
            return pd.DataFrame(columns=["rule", "support", "precision", "recall", "lift"])

        results = []
        y_mean = y.mean()

        for rule in rules:
            try:
                mask = X.eval(rule)
                if mask.sum() == 0:
                    continue
                precision = (y[mask].sum()) / mask.sum()
                recall = (y[mask].sum()) / y.sum() if y.sum() > 0 else 0
                lift = precision / (y_mean + 1e-9)
                results.append({
                    "rule": rule,
                    "support": int(mask.sum()),
                    "precision": round(float(precision), 4),
                    "recall": round(float(recall), 4),
                    "lift": round(float(lift), 4)
                })
            except Exception:
                continue

        if not results:
            return pd.DataFrame(columns=["rule", "support", "precision", "recall", "lift"])

        return pd.DataFrame(results).sort_values(by=["precision", "lift"], ascending=False).reset_index(drop=True)

    def export_rules(self, top_n: int = 10) -> List[str]:
        """
        Export top N rules.
        """
        if self.rules.empty:
            return []
        return self.rules.head(top_n)["rule"].tolist()

    def to_sql(self, rules: List[str]) -> str:
        """
        Convert rules into SQL WHERE clause.
        """
        if not rules:
            return "-- No valid rules found"
        sql_rules = []
        for rule in rules:
            sql_rule = rule.replace("and", "AND").replace("or", "OR")
            sql_rules.append(f"({sql_rule})")
        return " OR ".join(sql_rules)

if __name__ == "__main__":
    # ===== Demo with synthetic data =====
    np.random.seed(42)
    df = pd.DataFrame({
        "mcc_count": np.random.randint(0, 6, 1000),
        "cc_count": np.random.randint(0, 4, 1000),
        "procedure_count": np.random.randint(0, 8, 1000),
        "length_of_stay": np.random.randint(1, 15, 1000),
        "discharge_status": np.random.choice([1, 2, 3, 4, 5], 1000),
        "overpayment": np.random.choice([0, 1], 1000, p=[0.8, 0.2])
    })

    # Split features and target
    X = df.drop(columns="overpayment")
    y = df["overpayment"]

    # Fit miner
    miner = DRGRuleMiner(max_depth=4, min_samples_leaf=20)
    miner.fit(X, y)

    print("\n===== Top Rules =====")
    print(miner.rules.head() if not miner.rules.empty else "No rules found")

    print("\n===== SQL WHERE Clause =====")
    print(miner.to_sql(miner.export_rules()))
