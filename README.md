import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
import graphviz
import os
import numpy as np # For numerical operations

# --- 1. Create Dummy Data (REPLACE WITH YOUR ACTUAL X_train and y_train) ---
# In a real-world scenario, you would load your pre-split X_train and y_train here.
# For demonstration purposes, let's create a larger, more complex synthetic dataset
# that resembles DRG Medicare data with a bit more variety and potential for deeper splits.
# We'll also introduce some imbalance, as overpayment cases are typically rarer.

np.random.seed(42) # For reproducibility of dummy data

num_samples = 500 # Increased number of samples for a more complex tree

drg_codes = ['291', '292', '871', '872', '313', '470']
provider_types = ['Hospital', 'Skilled Nursing Facility', 'Ambulatory Surgical Center']
diagnosis_groups = ['Heart Failure', 'Pneumonia', 'Stroke', 'Diabetes', 'Kidney Failure', 'Cancer']

data = {
    'DRG_Code': np.random.choice(drg_codes, num_samples),
    'Provider_Type': np.random.choice(provider_types, num_samples),
    'Length_of_Stay': np.random.randint(1, 30, num_samples), # Days
    'Total_Charges': np.random.randint(5000, 100000, num_samples), # USD
    'Diagnosis_Group': np.random.choice(diagnosis_groups, num_samples),
    'Number_of_Procedures': np.random.randint(0, 5, num_samples),
    'Patient_Age_Group': np.random.choice(['Child', 'Adult', 'Senior'], num_samples),
    'Readmission_Flag': np.random.choice([0, 1], num_samples, p=[0.9, 0.1]), # 10% readmission
}
df = pd.DataFrame(data)

# Introduce some patterns for 'Overpayment' to make the target learnable
# This is a simplified logic for dummy data; real patterns would be more complex
df['Overpayment'] = 0
df.loc[(df['DRG_Code'] == '292') & (df['Length_of_Stay'] > 15) & (df['Total_Charges'] > 40000), 'Overpayment'] = 1
df.loc[(df['Provider_Type'] == 'Skilled Nursing Facility') & (df['Length_of_Stay'] > 20) & (df['Number_of_Procedures'] < 1), 'Overpayment'] = 1
df.loc[(df['Diagnosis_Group'] == 'Stroke') & (df['Total_Charges'] > 80000) & (df['Patient_Age_Group'] == 'Child'), 'Overpayment'] = 1
df.loc[df['Overpayment'] == 0, 'Overpayment'] = np.random.choice([0, 1], sum(df['Overpayment'] == 0), p=[0.98, 0.02]) # Add some random noise for complexity

# Ensure a minimum number of overpayment cases for demonstration
if df['Overpayment'].sum() < 20:
    # Force some more overpayments if the random generation was too sparse
    overpayment_indices = np.random.choice(df.index, 20 - df['Overpayment'].sum(), replace=False)
    df.loc[overpayment_indices, 'Overpayment'] = 1

print(f"Generated dummy data with {len(df)} samples. Overpayment cases: {df['Overpayment'].sum()}")
print(df.head())

# Separate features (X) and target (y)
X = df.drop('Overpayment', axis=1)
y = df['Overpayment']

# Split the data into training and testing sets. This is crucial for evaluating model generalization.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # stratify=y preserves class balance

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")
print(f"Overpayment in training set: {y_train.sum()} ({y_train.mean():.2%})")
print(f"Overpayment in test set: {y_test.sum()} ({y_test.mean():.2%})")


# --- 2. Identify Categorical and Numerical Features ---
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns

print(f"\nCategorical features: {list(categorical_features)}")
print(f"Numerical features: {list(numerical_features)}")

# --- 3. Preprocessing Pipeline for Categorical Variables (One-Hot Encoding) ---
# Create a column transformer to apply OneHotEncoder to categorical features.
# 'handle_unknown='ignore'' is important for robust deployment, handling categories not seen in training.
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough' # Keep numerical features as they are.
)

# --- 4. Build and Train the Decision Tree Classifier with Hyperparameter Tuning ---
# Create a pipeline that first preprocesses the data and then trains the model.
# We'll use GridSearchCV to find the best hyperparameters.

# Initial pipeline for GridSearchCV
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', DecisionTreeClassifier(random_state=42))])

# Define a broader parameter grid for tuning
# These ranges can be adjusted based on the size of your real dataset and initial experiments.
param_grid = {
    'classifier__max_depth': [5, 7, 10, 15, 20, None], # Max depth of the tree
    'classifier__min_samples_split': [2, 5, 10, 20], # Minimum samples required to split a node
    'classifier__min_samples_leaf': [1, 2, 5, 10],   # Minimum samples required at a leaf node
    'classifier__max_features': [None, 'sqrt', 'log2'], # Number of features to consider for best split
    'classifier__class_weight': [None, 'balanced'] # Handles imbalanced datasets by weighting classes
}

# Create GridSearchCV object
# cv=5 for 5-fold cross-validation.
# scoring='roc_auc' is often preferred for imbalanced classification.
# n_jobs=-1 uses all available CPU cores, speeding up the search.
# verbose=1 shows progress.
print("\nStarting Grid Search for best Decision Tree hyperparameters with 5-fold Cross-Validation...")
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)
print("Grid Search complete.")

# Get the best model found by GridSearchCV
best_model_pipeline = grid_search.best_estimator_
best_decision_tree_model = best_model_pipeline.named_steps['classifier']

print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Best ROC AUC score (on training folds): {grid_search.best_score_:.4f}")

# Get feature names after one-hot encoding for visualization and importance
# Fit the preprocessor on X_train to ensure it's ready for feature name extraction
preprocessor.fit(X_train)
encoded_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
# Combine one-hot encoded names with numerical feature names
all_feature_names = list(encoded_feature_names) + list(numerical_features)

# --- 5. Visualize the Best Decision Tree ---
# Export the best decision tree to a DOT file
dot_data = export_graphviz(best_decision_tree_model,
                           out_file=None,
                           feature_names=all_feature_names,
                           class_names=['No Overpayment', 'Overpayment'],
                           filled=True, rounded=True,
                           special_characters=True)

# Render the DOT file to a PDF
output_file_name = "drg_overpayment_decision_tree_tuned"
try:
    graph = graphviz.Source(dot_data)
    graph.render(output_file_name, format='pdf', view=False)
    print(f"\nDecision tree visualization saved to {output_file_name}.pdf")
except Exception as e:
    print(f"\nError rendering the decision tree visualization: {e}")
    print("Please ensure Graphviz is installed on your system and its executable is in your PATH.")
    print("You can download it from: https://graphviz.org/download/")
    print("And install the Python library: pip install graphviz")

# --- 6. Model Evaluation on the Test Set ---
print("\n" + "="*50)
print("--- Model Evaluation on Test Set ---")
print("="*50)

# Make predictions on the unseen test data using the best model
y_pred = best_model_pipeline.predict(X_test)
y_pred_proba = best_model_pipeline.predict_proba(X_test)[:, 1] # Probability of 'Overpayment' class

# Calculate and print various evaluation metrics
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
print(f"  True Negative (Predicted No Overpayment, Actual No Overpayment): {conf_matrix[0, 0]}")
print(f"  False Positive (Predicted Overpayment, Actual No Overpayment): {conf_matrix[0, 1]} (Type I Error - False Alert)")
print(f"  False Negative (Predicted No Overpayment, Actual Overpayment): {conf_matrix[1, 0]} (Type II Error - Missed Overpayment)")
print(f"  True Positive (Predicted Overpayment, Actual Overpayment): {conf_matrix[1, 1]}")

print("\nClassification Report (Precision, Recall, F1-Score):")
print(classification_report(y_test, y_pred, target_names=['No Overpayment', 'Overpayment']))

print(f"\nROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"Precision (Overpayment): {precision_score(y_test, y_pred):.4f}") # How many predicted overpayments were correct
print(f"Recall (Overpayment): {recall_score(y_test, y_pred):.4f}")     # How many actual overpayments were found
print(f"F1-Score (Overpayment): {f1_score(y_test, y_pred):.4f}")       # Balance between precision and recall

# --- 7. Feature Importance ---
print("\n" + "="*50)
print("--- Feature Importances ---")
print("="*50)

# Get feature importances from the best trained classifier
importances = best_decision_tree_model.feature_importances_

# Create a DataFrame for better readability and sort by importance
feature_importance_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print(feature_importance_df)
print("\nThese importances indicate which features the decision tree found most useful for making predictions.")

# --- 8. Guidance for Subject Matter Experts (SMEs) ---
print("\n" + "="*50)
print("--- Guidance for Subject Matter Experts (SMEs) ---")
print("="*50)
print("\nThe generated PDF file ('drg_overpayment_decision_tree_tuned.pdf') now shows a more optimized and potentially deeper decision tree.")
print("This tree was built using the best set of hyperparameters found through cross-validation, aiming for better generalization.")

print("\n### How to Interpret and Develop Concepts/Queries (Advanced Considerations):")
print("1.  **Analyze the Deeper Paths**: With a potentially larger tree, you'll find more nuanced rules. Focus on paths leading to 'Overpayment' leaves.")
print("    * **Complex Rule Example**: `IF DRG_Code_292 = 1 AND Length_of_Stay > 15.5 AND Total_Charges > 45000 AND Number_of_Procedures < 2 THEN Overpayment`.")
print("    * **Node Purity**: Observe the `value` array `[no_overpayment_count, overpayment_count]` in each node. A leaf node with a very skewed `value` towards `[0, N]` (meaning N overpayment cases and 0 no-overpayment cases) represents a 'purer' and stronger rule.")

print("\n2.  **Utilize Feature Importances**: The 'Feature Importances' table helps prioritize your investigation. Features with higher importance are more critical for identifying overpayments.")
print("    * Focus on the top-ranked features. Do these align with your domain expertise? Do they suggest new areas for investigation?")

print("\n3.  **Cross-Reference with Evaluation Metrics**:")
print("    * **Recall (Overpayment)**: If high, the model is good at catching actual overpayments. The rules derived from the tree are likely capturing many true cases.")
print("    * **Precision (Overpayment)**: If high, then when the model *predicts* overpayment, it's usually correct. This means the rules derived are quite specific and lead to fewer false alarms.")
print("    * **False Positives (Type I Error)**: These are cases where the model flags an overpayment, but it wasn't one. The associated tree paths might need further scrutiny or refinement by SMEs.")
print("    * **False Negatives (Type II Error)**: These are actual overpayments the model missed. Investigate these cases in your data to see if there are missing features or patterns the model couldn't capture, which could lead to further feature engineering.")

print("\n4.  **Refine SQL Queries for Auditing**: The more specific the tree rules, the more precise your SQL queries can be. This allows for highly targeted audits.")
print("    * **Example SQL with additional conditions from a deeper tree**:")
print("        ```sql")
print("        SELECT *")
print("        FROM Your_DRG_Medicare_Data")
print("        WHERE DRG_Code = '292'")
print("          AND Length_of_Stay > 15.5")
print("          AND Total_Charges > 45000")
print("          AND Number_of_Procedures < 2;")
print("        ```")

print("\n5.  **Develop Actionable Concepts**: Translate the quantitative rules into qualitative concepts for policy makers and auditors.")
print("    * **Concept**: \"Anomalously long stays (e.g., >15 days) for DRG 292 (Pneumonia) combined with high charges (e.g., >$45,000) and unusually few procedures (e.g., <2) are strong indicators of potential overpayment, possibly suggesting inflated billing or unnecessary services.\"\n")
print("    * This concept can then guide investigations and policy updates.")

print("\nBy iteratively analyzing the tree, evaluating its performance, and leveraging your domain knowledge, you can build a powerful system for identifying and preventing DRG Medicare overpayments.")
print("\nTo view the PDF, open 'drg_overpayment_decision_tree_tuned.pdf' in a PDF viewer.")
print("\n" + "="*50)

