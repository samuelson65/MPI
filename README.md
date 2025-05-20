import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dtreeviz import model

# === Example Data ===
data = {
    'age': [70, 55, 80, 45],
    'diagnosis': ['pneumonia', 'stroke', 'pneumonia', 'diabetes'],
    'procedure_code': ['XYZ1', 'ABC2', 'XYZ1', 'XYZ3'],
    'drg': ['193', '061', '193', '299']
}
df = pd.DataFrame(data)
X_train = df[['age', 'diagnosis', 'procedure_code']]
y_train = df['drg']

# === Step 1: Fit classifier ===
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train[['age']], y_train)  # dtreeviz needs numeric data

# === Step 2: dtreeviz v2.x model visualization ===
viz = model(clf,
            X_train=X_train[['age']],  # only numeric features supported directly
            y_train=y_train,
            feature_names=['age'],
            class_names=list(clf.classes_),
            target_name="DRG")

# === Step 3: Render view ===
viz.view()  # This opens in your default browser

# === Optional: Save SVG ===
viz.save("drg_decision_tree.svg")

# === Step 4: Rules in text ===
print("\n===== DECISION RULES =====")
print(export_text(clf, feature_names=['age']))
