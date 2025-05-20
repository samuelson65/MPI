import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dtreeviz.trees import dtreeviz
import matplotlib.pyplot as plt

# === Replace this with your real data ===
# Simulated example data
data = {
    'age': [70, 55, 80, 45],
    'diagnosis': ['pneumonia', 'stroke', 'pneumonia', 'diabetes'],
    'procedure_code': ['XYZ1', 'ABC2', 'XYZ1', 'XYZ3'],
    'drg': ['193', '061', '193', '299']
}
df = pd.DataFrame(data)

X_train = df[['age', 'diagnosis', 'procedure_code']]
y_train = df['drg']

# === Step 1: Identify feature types ===
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()

# === Step 2: Preprocessing and pipeline ===
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=4, random_state=42))
])

# === Step 3: Fit the model ===
pipeline.fit(X_train, y_train)

# === Step 4: Visualize tree interactively ===
clf = pipeline.named_steps['classifier']
preprocessor_fit = pipeline.named_steps['preprocessor']
X_transformed = preprocessor_fit.transform(X_train)
feature_names = preprocessor_fit.get_feature_names_out()

viz = dtreeviz(clf,
               X_train=X_transformed,
               y_train=y_train,
               feature_names=feature_names,
               class_names=list(clf.classes_),
               target_name="DRG")

viz.save("drg_decision_tree.svg")
print("Interactive tree saved as 'drg_decision_tree.svg'. You can open it in a browser.")

# === Step 5: Print text-based rules for SMEs ===
print("\n===== DECISION RULES =====\n")
rules_text = export_text(clf, feature_names=list(feature_names))
print(rules_text)
