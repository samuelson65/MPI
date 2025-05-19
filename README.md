import pandas as pd
from catboost import CatBoostClassifier, Pool
import graphviz

# Sample data
data = {
    'Provider_Type': ['Hospital', 'Clinic', 'Hospital', 'Private', 'Clinic'],
    'Procedure_Code': ['A100', 'B200', 'A100', 'C300', 'B200'],
    'Claim_Amount': [5000, 1200, 7500, 3000, 1500],
    'Patient_Age': [45, 32, 67, 29, 55],
    'Overpayment': [1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)
X = df.drop(columns=['Overpayment'])
y = df['Overpayment']

cat_features = ['Provider_Type', 'Procedure_Code']
pool = Pool(X, y, cat_features=cat_features, feature_names=list(X.columns))

# Train CatBoost model
final_model = CatBoostClassifier(
    iterations=10,
    depth=3,
    cat_features=cat_features,
    verbose=0
)
final_model.fit(pool)

# Export the first tree to dot format
dot_string = final_model.plot_tree(tree_idx=0, pool=pool, plot=False)

# Use graphviz to render and view the tree
graph = graphviz.Source(dot_string)
graph.render("decision_tree", format="png", cleanup=True)
graph.view("decision_tree")
