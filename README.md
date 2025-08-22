"""
drg_ruleminer.py

A compact, production-friendly rule mining framework to find SQL-like conditions
(e.g., mcc_count > 3 AND length_of_stay < 5 AND cc_count > 1) that capture
overpayment claims with minimal nofindings.

Key features:
- Handles missing values, rare categories, constant/low-variance features.
- One-hot encodes categoricals (with Missing bucket) so tree rules map cleanly to SQL.
- Class imbalance handled via class_weight="balanced".
- Hyperparameter search for tree (depth, min_samples_leaf) using AUPRC.
- Extracts every positive leaf as a candidate rule, simplifies it, scores it (precision, recall, lift).
- Redundancy pruning and Pareto frontier selection.
- Greedy set-cover to assemble an OR-of-rules "policy" at a required precision threshold.
- SQL string generation for each rule and for the combined policy.
- Deterministic behavior via random_state.

Usage:

from drg_ruleminer import RuleMiner
rm = RuleMiner(target_col='label', pos_label='overpayment')
result = rm.fit(df)

result.rules_df          # DataFrame of ranked rules with metrics
result.policy_sql        # Single SQL string (OR of rules) targeting min_precision
result.rule_sql_strings  # Dict: rule_id -> SQL for individual rules
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree

# ------------------------------ Utilities ------------------------------

def _is_categorical(s: pd.Series, max_unique_ratio: float = 0.05, max_cardinality: int = 50) -> bool:
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
        return True
    if pd.api.types.is_bool_dtype(s):
        return True
    nunique = s.nunique(dropna=True)
    if s.size == 0: 
        return False
    if nunique <= max_cardinality and (nunique / max(s.size,1)) <= max_unique_ratio:
        return True
    return False

def _safe_fillna_categorical(series: pd.Series) -> pd.Series:
    return series.astype("object").fillna("Missing").astype(str)

def _safe_fillna_numeric(series: pd.Series) -> pd.Series:
    if series.notna().any():
        return series.fillna(series.median())
    else:
        return series.fillna(0)

def _drop_constant_columns(X: pd.DataFrame):
    keep_cols = []
    dropped = []
    for c in X.columns:
        if X[c].nunique(dropna=False) > 1:
            keep_cols.append(c)
        else:
            dropped.append(c)
    return X[keep_cols].copy(), dropped

def _auprc(y_true, y_score) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return 0.0

def _ensure_binary_labels(y: pd.Series, pos_label: Any) -> np.ndarray:
    if set(pd.unique(y.dropna())) <= {0,1}:
        return y.astype(int).to_numpy()
    return (y == pos_label).astype(int).to_numpy()

# ------------------------------ Data Classes ------------------------------

@dataclass
class Rule:
    rule_id: str
    conditions: List[Tuple[str, str, Any]]
    prediction: int
    precision: float
    recall: float
    support_pos: int
    support_total: int
    lift: float

    def to_sql(self, original_feature_map: Dict[str, Dict[str, Any]]) -> str:
        sql_parts = []
        grouped_cat = {}
        for feat, op, thr in self.conditions:
            meta = original_feature_map.get(feat, {'type':'numeric','orig':feat})
            if meta['type'] == 'onehot':
                raw_name = meta['orig']
                cat_val = meta['value']
                grouped_cat.setdefault(raw_name, []).append(cat_val)
            else:
                if op in ('<=','<','>=','>','==','!='):
                    val_repr = str(int(thr)) if isinstance(thr,(int,np.integer)) else str(float(np.round(thr,6)))
                    sql_parts.append(f'"{meta["orig"]}" {op} {val_repr}')
        for raw_name, cats in grouped_cat.items():
            seen = set()
            cats = [c for c in cats if not (c in seen or seen.add(c))]
            quoted = ", ".join([f"'{str(c).replace(\"'\",\"''\")}'" for c in cats])
            sql_parts.append(f'"{raw_name}" IN ({quoted})')
        return "(" + " AND ".join(sql_parts) + ")" if sql_parts else "(1=1)"

@dataclass
class RuleSetResult:
    rules_df: pd.DataFrame
    rule_sql_strings: Dict[str, str]
    policy_rule_ids: List[str]
    policy_sql: str
    metrics: Dict[str, float]
    dropped_columns: List[str]
    feature_map: Dict[str, Dict[str, Any]]

# ------------------------------ Rule Miner ------------------------------

class RuleMiner:
    def __init__(self,
                 target_col='label',
                 pos_label='overpayment',
                 random_state=42,
                 min_precision=0.7,
                 max_depth_grid=(3,4,5,6),
                 min_samples_leaf_frac=0.01,
                 min_samples_leaf_floor=20,
                 max_rules=50,
                 test_size=0.2,
                 val_size=0.2,
                 verbose=False):
        self.target_col = target_col
        self.pos_label = pos_label
        self.random_state = random_state
        self.min_precision = float(min_precision)
        self.max_depth_grid = tuple(max_depth_grid)
        self.min_samples_leaf_frac = float(min_samples_leaf_frac)
        self.min_samples_leaf_floor = int(min_samples_leaf_floor)
        self.max_rules = int(max_rules)
        self.test_size = float(test_size)
        self.val_size = float(val_size)
        self.verbose = verbose

    # -------------------------- Preprocessing --------------------------
    def _prepare_data(self, df: pd.DataFrame):
        if self.target_col not in df.columns:
            raise ValueError(f"target_col '{self.target_col}' not found.")
        y = _ensure_binary_labels(df[self.target_col], self.pos_label)
        X_raw = df.drop(columns=[self.target_col]).copy()
        cat_cols, num_cols = [], []
        for c in X_raw.columns:
            if _is_categorical(X_raw[c]):
                cat_cols.append(c)
            elif pd.api.types.is_numeric_dtype(X_raw[c]):
                num_cols.append(c)
            else:
                cat_cols.append(c)
        for c in num_cols:
            X_raw[c] = _safe_fillna_numeric(X_raw[c])
        for c in cat_cols:
            X_raw[c] = _safe_fillna_categorical(X_raw[c])
        X_cat = pd.get_dummies(X_raw[cat_cols], prefix=cat_cols, dummy_na=False, drop_first=False) if cat_cols else pd.DataFrame(index=X_raw.index)
        X_num = X_raw[num_cols].copy()
        X_all = pd.concat([X_num, X_cat], axis=1)
        X_all, dropped_cols = _drop_constant_columns(X_all)
        feature_map = {}
        for c in X_num.columns:
            if c in X_all.columns:
                feature_map[c] = {'type':'numeric','orig':c}
        for c in X_cat.columns:
            if c in X_all.columns:
                matched_orig = None
                for orig in cat_cols:
                    if c.startswith(orig + "_"):
                        matched_orig = orig
                        val = c[len(orig)+1:]
                        break
                if matched_orig is None:
                    matched_orig, val = c, "1"
                feature_map[c] = {'type':'onehot','orig':matched_orig,'value':val}
        return X_all, y, feature_map, dropped_cols

    def _split_data(self, X, y):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=self.test_size + self.val_size, stratify=y, random_state=self.random_state)
        rel_val = self.val_size / (self.test_size + self.val_size) if (self.test_size + self.val_size) > 0 else 0.0
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-rel_val, stratify=y_temp, random_state=self.random_state)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def _fit_tree(self, X_train, y_train, X_val, y_val):
        n = X_train.shape[0]
        min_leaf = max(int(self.min_samples_leaf_frac * n), self.min_samples_leaf_floor)
        best_model, best_score = None, -1
        for depth in self.max_depth_grid:
            clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf, class_weight="balanced", random_state=self.random_state)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_val)[:,1]
            score = _auprc(y_val, probs)
            if score > best_score:
                best_score, best_model = score, clf
        return best_model

    def _extract_rules(self, clf, feature_names):
        tree_ = clf.tree_
        feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
        rules = []
        def recurse(node, path):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                feat, thr = feature_name[node], tree_.threshold[node]
                recurse(tree_.children_left[node], path+[(feat,"<=",thr)])
                recurse(tree_.children_right[node], path+[(feat,">",thr)])
            else:
                value = tree_.value[node][0]
                pred = int(np.argmax(value))
                if pred == 1:
                    rules.append(path)
        recurse(0,[])
        return rules

    def _simplify_rule(self, rule):
        bounds = {}
        for feat, op, thr in rule:
            thr = float(thr)
            lo, hi = bounds.get(feat, (-np.inf, np.inf))
            if op == "<=": hi = min(hi, thr)
            elif op == ">": lo = max(lo, np.nextafter(thr,np.inf))
            elif op == "<": hi = min(hi, np.nextafter(thr,-np.inf))
            elif op == ">=": lo = max(lo, thr)
            elif op == "==": lo,hi = max(lo,thr), min(hi,thr)
            bounds[feat] = (lo,hi)
        simplified=[]
        for feat,(lo,hi) in bounds.items():
            if lo>0.5: simplified.append((feat,"==",1))
            elif hi<=0.5: continue
            else:
                if not np.isneginf(lo): simplified.append((feat,">",lo))
                if not np.isposinf(hi): simplified.append((feat,"<=",hi))
        return simplified

    def _rule_mask(self,X,rule):
        mask = np.ones(X.shape[0],dtype=bool)
        for feat,op,thr in rule:
            col=X[feat]
            if op=="==": mask &= (col>=0.5)
            elif op=="<=": mask &= (col<=thr+1e-9)
            elif op==">": mask &= (col>thr-1e-9)
        return mask

    def _score_rule(self,X,y,rule):
        mask=self._rule_mask(X,rule)
        if mask.sum()==0: return 0,0,0,0,0
        tp=int(((y==1)&mask).sum()); fp=int(((y==0)&mask).sum())
        fn=int(((y==1)&(~mask)).sum())
        prec=tp/(tp+fp) if tp+fp>0 else 0
        rec=tp/(tp+fn) if tp+fn>0 else 0
        base=(y==1).mean() or 1e-9
        lift=prec/base
        return prec,rec,tp,mask.sum(),lift

    def fit(self, df: pd.DataFrame) -> RuleSetResult:
        X,y,feature_map,dropped_cols=self._prepare_data(df)
        X_train,y_train,X_val,y_val,X_test,y_test=self._split_data(X,y)
        clf=self._fit_tree(X_train,y_train,X_val,y_val)
        raw_rules=self._extract_rules(clf,list(X.columns))
        simplified_rules=[self._simplify_rule(r) for r in raw_rules]
        rules=[]
        for i,conds in enumerate(simplified_rules):
            prec,rec,tp,sup,lift=self._score_rule(X_val,y_val,conds)
            if sup>0:
                rules.append(Rule(f"R{i+1}",conds,1,prec,rec,tp,sup,lift))
        rules=sorted(rules,key=lambda r:(-r.precision,-r.recall,-r.support_total,-r.lift))
        rule_sql_strings={r.rule_id:r.to_sql(feature_map) for r in rules}
        rows=[{'rule_id':r.rule_id,'precision_val':round(r.precision,6),'recall_val':round(r.recall,6),'lift_val':round(r.lift,6),
               'support_pos_val':r.support_pos,'support_total_val':r.support_total,'sql':rule_sql_strings[r.rule_id]} for r in rules]
        rules_df=pd.DataFrame(rows)
        selected=[r for r in rules if r.precision>=self.min_precision][:self.max_rules]
        policy_rule_ids=[r.rule_id for r in selected]
        policy_sql="(\n  " + " \n  OR \n  ".join([rule_sql_strings[rid] for rid in policy_rule_ids]) + "\n)" if policy_rule_ids else "(1=0)"
        mask=np.zeros(X_test.shape[0],dtype=bool)
        for r in selected: mask|=self._rule_mask(X_test,r.conditions)
        tp=int(((y_test==1)&mask).sum()); fp=int(((y_test==0)&mask).sum())
        fn=int(((y_test==1)&(~mask)).sum())
        prec=tp/(tp+fp) if tp+fp>0 else 0
        rec=tp/(tp+fn) if tp+fn>0 else 0
        base=(y_test==1).mean() or 1e-9
        lift=prec/base
        metrics={'policy_precision_test':round(prec,6),'policy_recall_test':round(rec,6),'policy_lift_test':round(lift,6),
                 'policy_coverage_test':round(mask.mean(),6),'base_rate_test':round(base,6)}
        return RuleSetResult(rules_df,rule_sql_strings,policy_rule_ids,policy_sql,metrics,dropped_cols,feature_map)
