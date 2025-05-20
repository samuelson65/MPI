import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dtreeviz import model
import numpy as np

# === Sample DRG-style data ===
data = {
    'age': [70, 55, 80, 45],
    'diagnosis': ['pneumonia', 'stroke', 'pneumonia', 'diabetes'],
    'procedure_code': ['XYZ1', 'ABC2', 'XYZ1', 'XYZ3'],
    'drg': ['193', '061', '193', '299']
}
df = pd.DataFrame(data)

X_train = df[['age', 'diagnosis', 'procedure_code']]
y_train = df['drg']

# === Step 1: Preprocessing ===
categorical_cols = ['diagnosis', 'procedure_code']
numerical_cols = ['age']

# OneHotEncoder will output float32, no strings
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numerical_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=4, random_state=42))
])

# === Step 2: Train pipeline ===
pipeline.fit(X_train, y_train)

# === Step 3: Extract parts for dtreeviz ===
clf = pipeline.named_steps['classifier']
X_encoded = pipeline.named_steps['preprocessor'].transform(X_train)

# Get clean numeric feature names
ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
cat_feature_names = ohe.get_feature_names_out(categorical_cols)
feature_names = list(cat_feature_names) + numerical_cols

X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names)

# === Step 4: DTreeViz v2.x visualization ===
viz = model(clf,
            X_train=X_encoded_df,
            y_train=y_train,
            feature_names=feature_names,
            class_names=clf.classes_,
            target_name="DRG")

# Correct method in v2.x
viz_obj = viz.view()
viz_obj.save("drg_decision_tree.svg")

print("Tree visualization saved as 'drg_decision_tree.svg'.")

# === Step 5: Print text rules ===
print("\n===== DECISION RULES =====")
print(export_text(clf, feature_names=feature_names))
