import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imodels.rule_sets.rulefit import RuleFit
from sklearn.tree import export_graphviz
import graphviz

# Step 1: Create sample categorical dataset
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'DRG': np.random.choice(['194', '470', '329', '870'], size=n),
    'PDX_MDC': np.random.choice(['01', '05', '08', '10'], size=n),
    'PROC_MDC': np.random.choice(['01', '05', '08', '10'], size=n),
    'SEX': np.random.choice(['M', 'F'], size=n),
    'AGE_GROUP': np.random.choice(['0-17', '18-34', '35-49', '50-64', '65+'], size=n)
})

# Step 2: Create binary target label
df['OVERPAID'] = (
    (df['DRG'].isin(['194', '470'])) &
    (df['PDX_MDC'] != df['PROC_MDC']) &
    (df['AGE_GROUP'].isin(['65+', '50-64']))
).astype(int)

# Step 3: Prepare data
X = pd.get_dummies(df.drop(columns='OVERPAID'), drop_first=True)
y = df['OVERPAID']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Step 4: Train RuleFit model
model = RuleFit(max_rules=10)
model.fit(X_train.values, y_train.values, feature_names=X.columns)

# Step 5: Get and print extracted rules
rules_df = model.get_rules()
rules_df = rules_df[rules_df.coef != 0].sort_values(by='support', ascending=False)
print("Top rules:\n", rules_df[['rule', 'coef', 'support']].head())

# Step 6: Visualize the first base decision tree
tree = model.tree_generator.estimators_[0]
dot_data = export_graphviz(
    tree,
    feature_names=X.columns,
    class_names=['Not Overpaid', 'Overpaid'],
    filled=True,
    rounded=True,
    out_file=None
)
graph = graphviz.Source(dot_data)
graph.render("rulefit_tree", format="png", cleanup=True)
graph.view("rulefit_tree")


imodels==1.3.4
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.26.4
graphviz==0.20.1

pip install -r requirements.txt
