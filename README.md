import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dtreeviz import dtreeviz

# === Example Input Data ===
data = {
    'age': [70, 55, 80, 45],
    'diagnosis': ['pneumonia', 'stroke', 'pneumonia', 'diabetes'],
    'procedure_code': ['XYZ1', 'ABC2', 'XYZ1', 'XYZ3'],
    'drg': ['193', '061', '193', '299']
}
df = pd.DataFrame(data)
X_train = df[['age', 'diagnosis', 'procedure_code']]
y_train = df['drg']

# === Step 1: Preprocessing for model training ===
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numerical_cols)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=4, random_state=42))
])

# === Step 2: Train pipeline ===
pipeline.fit(X_train, y_train)

# === Step 3: Access classifier from pipeline ===
clf = pipeline.named_steps['classifier']

# === Step 4: Visualize with dtreeviz using RAW X_train ===
viz = dtreeviz(clf,
               X_train=X_train,
               y_train=y_train,
               feature_names=X_train.columns,
               class_names=list(clf.classes_),
               target_name="DRG")

viz.save("drg_decision_tree.svg")
print("Interactive tree saved as 'drg_decision_tree.svg'. Open it in a browser.")

# === Step 5: Print rules ===
print("\n===== DECISION RULES FOR SME UNDERSTANDING =====\n")
print(export_text(clf, feature_names=list(X_train.columns)))
