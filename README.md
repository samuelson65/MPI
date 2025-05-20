import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dtreeviz import model
import numpy as np

# === Sample Data ===
data = {
    'age': [70, 55, 80, 45],
    'diagnosis': ['pneumonia', 'stroke', 'pneumonia', 'diabetes'],
    'procedure_code': ['XYZ1', 'ABC2', 'XYZ1', 'XYZ3'],
    'drg': ['193', '061', '193', '299']
}
df = pd.DataFrame(data)

# === Define X and y ===
X_train = df[['age', 'diagnosis', 'procedure_code']]
y_train = df['drg']

# === Preprocessing ===
categorical_cols = ['diagnosis', 'procedure_code']
numerical_cols = ['age']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numerical_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=4, random_state=42))
])

# === Train ===
pipeline.fit(X_train, y_train)
clf = pipeline.named_steps['classifier']
X_encoded = pipeline.named_steps['preprocessor'].transform(X_train)

# === Feature Names ===
ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
cat_feature_names = ohe.get_feature_names_out(categorical_cols)
feature_names = list(cat_feature_names) + numerical_cols

# === DTreeViz visualization ===
X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names)

viz = model(clf,
            X_train=X_encoded_df,
            y_train=y_train,
            feature_names=feature_names,
            class_names=clf.classes_,
            target_name="DRG")

viz_obj = viz.view()
viz_obj.save("drg_decision_tree.svg")  # Correct method in dtreeviz v2.x

print("SVG saved as 'drg_decision_tree.svg'.")

# === Optional: Text-based Rules ===
print("\n===== DECISION RULES (for SME) =====")
print(export_text(clf, feature_names=feature_names))
