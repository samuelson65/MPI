import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from typing import List, Tuple


class DRGRuleMiner:
    def __init__(self, max_depth=4, min_samples_leaf=50, class_weight="balanced"):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.tree = None
        self.feature_names = None
        self.rules = []

    def _tree_to_rules(self, tree, feature_names):
        rules = []

        def recurse(node, path):
            if tree.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_names[tree.feature[node]]
                threshold = tree.threshold[node]
                recurse(tree.children_left[node], path + [f"({name} <= {threshold:.2f})"])
                recurse(tree.children_right[node], path + [f"({name} > {threshold:.2f})"])
            else:
                rules.append(" AND ".join(path))

        recurse(0, [])
        return rules

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = list(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        self.tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=42
        )
        self.tree.fit(X_train, y_train)

        candidate_rules = self._tree_to_rules(self.tree.tree_, self.feature_names)
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
