import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

# Step 1: Create categorical dataset
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'DRG': np.random.choice(['194', '470', '329', '870'], size=n),
    'PDX_MDC': np.random.choice(['01', '05', '08', '10'], size=n),
    'PROC_MDC': np.random.choice(['01', '05', '08', '10'], size=n),
    'SEX': np.random.choice(['M', 'F'], size=n),
    'AGE_GROUP': np.random.choice(['0-17', '18-34', '35-49', '50-64', '65+'], size=n)
})

# Step 2: Create label
df['OVERPAID'] = (
    (df['DRG'].isin(['194', '470'])) &
    (df['PDX_MDC'] != df['PROC_MDC']) &
    (df['AGE_GROUP'].isin(['65+', '50-64']))
).astype(int)

# Step 3: Encode features
X = pd.get_dummies(df.drop(columns='OVERPAID'), drop_first=True)
y = df['OVERPAID']

# Step 4: Train Decision Tree
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X, y)

# Step 5: Visualize the tree
dot_data = export_graphviz(
    clf,
    feature_names=X.columns,
    class_names=["Not Overpaid", "Overpaid"],
    filled=True,
    rounded=True,
    out_file=None
)
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="png", cleanup=True)
graph.view("decision_tree")
