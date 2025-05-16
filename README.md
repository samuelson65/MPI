import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imodels import RuleFitClassifier
from sklearn.tree import export_graphviz
import graphviz

# Step 1: Create Categorical-Only DataFrame
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'DRG': np.random.choice(['194', '470', '329', '870'], size=n),
    'PDX_MDC': np.random.choice(['01', '05', '08', '10'], size=n),
    'PROC_MDC': np.random.choice(['01', '05', '08', '10'], size=n),
    'SEX': np.random.choice(['M', 'F'], size=n),
    'AGE_GROUP': np.random.choice(['0-17', '18-34', '35-49', '50-64', '65+'], size=n)
})

# Step 2: Generate Binary Label
df['OVERPAID'] = (
    (df['DRG'].isin(['194', '470'])) &
    (df['PDX_MDC'] != df['PROC_MDC']) &
    (df['AGE_GROUP'].isin(['65+', '50-64']))
).astype(int)

# Step 3: Encode categorical features
X = pd.get_dummies(df.drop('OVERPAID', axis=1), drop_first=True)
y = df['OVERPAID']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Step 4: Train RuleFit Model
model = RuleFitClassifier(max_rules=10)
model.fit(X_train, y_train)

# Step 5: Visualize one base decision tree
tree = model.rule_ensemble_.estimators_[0]
dot_data = export_graphviz(
    tree,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    out_file=None
)
graph = graphviz.Source(dot_data)
graph.render("rulefit_tree", format="png", cleanup=True)
graph.view("rulefit_tree")
