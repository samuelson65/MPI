import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imodels import RuleFitClassifier
from sklearn.tree import export_graphviz
import graphviz

# 1. Create sample categorical data
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'DRG': np.random.choice(['194', '470', '329', '870'], size=n),
    'PDX_MDC': np.random.choice(['01', '05', '08', '10'], size=n),
    'PROC_MDC': np.random.choice(['01', '05', '08', '10'], size=n),
    'SEX': np.random.choice(['M', 'F'], size=n),
    'AGE_GROUP': np.random.choice(['0-17', '18-34', '35-49', '50-64', '65+'], size=n)
})

# 2. Create binary label
df['OVERPAID'] = (
    (df['DRG'].isin(['194', '470'])) &
    (df['PDX_MDC'] != df['PROC_MDC']) &
    (df['AGE_GROUP'].isin(['65+', '50-64']))
).astype(int)

# 3. Encode features
X = pd.get_dummies(df.drop(columns='OVERPAID'), drop_first=True)
y = df['OVERPAID']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 4. Train RuleFit model
model = RuleFitClassifier(max_rules=10)
model.fit(X_train, y_train)

# 5. Get and display top rules
rules_df = model.get_rules()
rules_df = rules_df[rules_df['coefficient'] != 0].sort_values(by='importance', ascending=False)
print("Top rules:\n", rules_df[['rule', 'support', 'importance']].head())

# 6. Visualize one source decision tree used by RuleFit (if available)
# NOTE: RuleFit uses trees from sklearn.ensemble.RandomForestRegressor by default
tree_model = model.model  # this is the underlying RandomForestRegressor
first_tree = tree_model.estimators_[0]

# Generate tree graph
dot_data = export_graphviz(
    first_tree,
    feature_names=X.columns,
    class_names=['Not Overpaid', 'Overpaid'],
    filled=True,
    rounded=True,
    out_file=None
)
graph = graphviz.Source(dot_data)
graph.render("rulefit_tree", format="png", cleanup=True)
graph.view("rulefit_tree")
