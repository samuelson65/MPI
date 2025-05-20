import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dtreeviz import dtreeviz
import numpy as np

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

# === Step 1: Define Categorical and Numerical Columns ===
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()

# === Step 2: Preprocessing and Model Pipeline ===
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numerical_cols)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=4, random_state=42))
])

# === Step 3: Fit the Model ===
pipeline.fit(X_train, y_train)

# === Step 4: Visualize the Tree Interactively ===
clf = pipeline.named_steps['classifier']
preprocessor_fit = pipeline.named_steps['preprocessor']

# Transform X and convert to DataFrame with proper column names
X_transformed = preprocessor_fit.transform(X_train)
feature_names = preprocessor_fit.get_feature_names_out()
X_transformed_df = pd.DataFrame(X_transformed.toarray() if hasattr(X_transformed, 'toarray') else X_transformed,
                                columns=feature_names)

# dtreeviz visualization
viz = dtreeviz(clf,
               X_train=X_transformed_df,
               y_train=y_train,
               feature_names=feature_names,
               class_names=list(clf.classes_),
               target_name="DRG")

viz.save("drg_decision_tree.svg")
print("Interactive tree saved as 'drg_decision_tree.svg'. Open it in a browser.")

# === Step 5: Print Readable Rules ===
print("\n===== DECISION RULES FOR SME UNDERSTANDING =====\n")
print(export_text(clf, feature_names=list(feature_names)))
