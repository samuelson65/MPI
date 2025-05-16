import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imodels.rule_sets.rulefit import RuleFit
from sklearn.tree import export_graphviz
import graphviz

# Create categorical dataset
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'DRG': np.random.choice(['194', '470', '329', '870'], size=n),
    'PDX_MDC': np.random.choice(['01', '05', '08', '10'], size=n),
    'PROC_MDC': np.random.choice(['01', '05', '08', '10'], size=n),
    'SEX': np.random.choice(['M', 'F'], size=n),
    'AGE_GROUP': np.random.choice(['0-17', '18-34', '35-49', '50-64', '65+'], size=n)
})

# Generate target
df['OVERPAID'] = (
    (df['DRG'].isin(['194', '470'])) &
    (df['PDX_MDC'] != df['PROC_MDC']) &
    (df['AGE_GROUP'].isin(['65+', '50-64']))
).astype(int)

# Prepare features
X = pd.get_dummies(df.drop(columns='OVERPAID'), drop_first=True)
y = df['OVERPAID']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train RuleFit model
model = RuleFit(max_rules=10)
model.fit(X_train.values, y_train.values, feature_names=X.columns)

# Get and print rules
rules_df = model.get_rules()
rules_df = rules_df[rules_df.coef != 0].sort_values(by='support', ascending=False)
print(rules_df[['rule', 'coef', 'support']].head())

# Visualize first decision tree used to create rules
base_tree = model.tree_generator.estimators_[0]
dot_data = export_graphviz(
    base_tree,
    feature_names=X.columns,
    class_names=['Not Overpaid', 'Overpaid'],
    filled=True,
    rounded=True,
    out_file=None
)
graph = graphviz.Source(dot_data)
graph.render("rulefit_tree", format="png", cleanup=True)
graph.view("rulefit_tree")
