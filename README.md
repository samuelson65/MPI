import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
import graphviz
import os
import numpy as np

# --- 1. Create Dummy Data (REPLACE WITH YOUR ACTUAL X_train and y_train) ---
np.random.seed(42) # For reproducibility of dummy data

num_samples = 1000 # Increased number of samples for a more complex tree

drg_codes = ['291', '292', '871', '872', '313', '470', '601', '789']
provider_types = ['Hospital', 'Skilled Nursing Facility', 'Ambulatory Surgical Center', 'Rehab Center']
diagnosis_groups = ['Heart Failure', 'Pneumonia', 'Stroke', 'Diabetes', 'Kidney Failure', 'Cancer', 'Fracture', 'Sepsis']
patient_age_groups = ['Child', 'Young Adult', 'Adult', 'Senior']
payer_types = ['Medicare', 'Medicaid', 'Commercial', 'Self-Pay']

data = {
    'DRG_Code': np.random.choice(drg_codes, num_samples),
    'Provider_Type': np.random.choice(provider_types, num_samples),
    'Length_of_Stay': np.random.randint(1, 40, num_samples), # Days
    'Total_Charges': np.random.randint(5000, 150000, num_samples), # USD
    'Diagnosis_Group': np.random.choice(diagnosis_groups, num_samples),
    'Number_of_Procedures': np.random.randint(0, 7, num_samples),
    'Patient_Age_Group': np.random.choice(patient_age_groups, num_samples),
    'Readmission_Flag': np.random.choice([0, 1], num_samples, p=[0.95, 0.05]), # 5% readmission
    'Payer_Type': np.random.choice(payer_types, num_samples),
    'Comorbidity_Index': np.random.randint(0, 5, num_samples) # Example of a numerical index
}
df = pd.DataFrame(data)

# Introduce more complex patterns for 'Overpayment' to make the target learnable and diverse
df['Overpayment'] = 0
# Pattern 1: High charges, short stay for specific DRG
df.loc[(df['DRG_Code'] == '292') & (df['Length_of_Stay'] < 5) & (df['Total_Charges'] > 70000), 'Overpayment'] = 1
# Pattern 2: Very long stay, low procedures for certain provider type
df.loc[(df['Provider_Type'] == 'Skilled Nursing Facility') & (df['Length_of_Stay'] > 30) & (df['Number_of_Procedures'] < 2), 'Overpayment'] = 1
# Pattern 3: Specific diagnosis with high charges for seniors
df.loc[(df['Diagnosis_Group'] == 'Stroke') & (df['Total_Charges'] > 100000) & (df['Patient_Age_Group'] == 'Senior'), 'Overpayment'] = 1
# Pattern 4: High comorbidity index with very low charges (might indicate under-billing in some cases, but could be over for others due to miscoding)
df.loc[(df['Comorbidity_Index'] > 3) & (df['Total_Charges'] < 10000), 'Overpayment'] = 1
# Pattern 5: Readmission with high charges for specific DRG
df.loc[(df['Readmission_Flag'] == 1) & (df['DRG_Code'] == '470') & (df['Total_Charges'] > 50000), 'Overpayment'] = 1

# Add some random noise for complexity, keeping the target imbalanced
df.loc[df['Overpayment'] == 0, 'Overpayment'] = np.random.choice([0, 1], sum(df['Overpayment'] == 0), p=[0.97, 0.03])

# Ensure a minimum number of overpayment cases for demonstration purposes
if df['Overpayment'].sum() < 30: # Aim for at least 30 overpayment cases
    missing_overpayments = 30 - df['Overpayment'].sum()
    if missing_overpayments > 0:
        overpayment_indices = np.random.choice(df.index[df['Overpayment'] == 0], missing_overpayments, replace=False)
        df.loc[overpayment_indices, 'Overpayment'] = 1


print(f"Generated dummy data with {len(df)} samples. Overpayment cases: {df['Overpayment'].sum()}")
print(df.head())

# Separate features (X) and target (y)
X = df.drop('Overpayment', axis=1)
y = df['Overpayment']

# Split the data into training and testing sets. Stratify to preserve class balance.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

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
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

# --- 4. Build and Train the Decision Tree Classifier with Hyperparameter Tuning ---
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', DecisionTreeClassifier(random_state=42))])

# Define a broader parameter grid for tuning
param_grid = {
    'classifier__max_depth': [7, 10, 15, 20], # Test different max depths
    'classifier__min_samples_split': [5, 10, 20], # Minimum samples required to split a node
    'classifier__min_samples_leaf': [3, 5, 10],   # Minimum samples required at a leaf node
    'classifier__max_features': [None, 'sqrt', 'log2'], # Number of features to consider for best split
    'classifier__class_weight': [None, 'balanced'] # Handles imbalanced datasets by weighting classes
}

print("\nStarting Grid Search for best Decision Tree hyperparameters with 5-fold Cross-Validation...")
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)
print("Grid Search complete.")

best_model_pipeline = grid_search.best_estimator_
best_decision_tree_model = best_model_pipeline.named_steps['classifier']

print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Best ROC AUC score (on training folds): {grid_search.best_score_:.4f}")

# Get feature names after one-hot encoding for visualization and importance
preprocessor.fit(X_train)
encoded_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = list(encoded_feature_names) + list(numerical_features)

# --- 5. Visualize the Best Decision Tree ---
dot_data = export_graphviz(best_decision_tree_model,
                           out_file=None,
                           feature_names=all_feature_names,
                           class_names=['No Overpayment', 'Overpayment'],
                           filled=True, rounded=True,
                           special_characters=True)

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

y_pred = best_model_pipeline.predict(X_test)
y_pred_proba = best_model_pipeline.predict_proba(X_test)[:, 1]

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
print(f"Precision (Overpayment): {precision_score(y_test, y_pred):.4f}")
print(f"Recall (Overpayment): {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score (Overpayment): {f1_score(y_test, y_pred):.4f}")

# --- 7. Feature Importance ---
print("\n" + "="*50)
print("--- Feature Importances ---")
print("="*50)

importances = best_decision_tree_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print(feature_importance_df)

# --- 8. Actionable Concept Generation ---
print("\n" + "="*50)
print("--- Actionable Overpayment Concepts (Derived from Decision Tree Rules) ---")
print("="*50)

# Function to extract rules from the decision tree
# Now stores feature_idx, operator_type (left/right split), and threshold
def get_tree_rules(tree_model, feature_names, class_names):
    tree_ = tree_model.tree_
    rules = []

    def recurse(node, path_conditions):
        if tree_.feature[node] != -2: # Not a leaf node
            feature_idx = tree_.feature[node]
            threshold = tree_.threshold[node]

            # Left child (True condition: feature <= threshold)
            recurse(tree_.children_left[node], path_conditions + [(feature_idx, '<=', threshold)])

            # Right child (False condition: feature > threshold)
            recurse(tree_.children_right[node], path_conditions + [(feature_idx, '>', threshold)])
        else: # Leaf node
            # Check if this leaf predicts 'Overpayment' (class_id = 1)
            # The value array is [samples_class_0, samples_class_1]
            if np.argmax(tree_.value[node]) == 1: # If the majority class is 'Overpayment'
                num_overpayment_samples = tree_.value[node][0][1] # Get count of overpayment samples at this leaf
                total_samples_at_leaf = tree_.n_node_samples[node]

                # Only consider rules that lead to a "pure enough" overpayment leaf
                # and cover a reasonable number of samples to be actionable.
                # Adjust these thresholds as needed for your data.
                if num_overpayment_samples > 0 and total_samples_at_leaf >= 5: # At least 5 samples at leaf
                    purity = num_overpayment_samples / total_samples_at_leaf
                    if purity >= 0.7: # At least 70% of samples at this leaf are overpayments
                        rules.append({
                            'conditions': path_conditions, # Store as (feature_idx, operator, threshold)
                            'predicted_class': class_names[np.argmax(tree_.value[node])],
                            'overpayment_samples': int(num_overpayment_samples),
                            'total_samples_at_leaf': int(total_samples_at_leaf),
                            'purity': purity
                        })

    recurse(0, []) # Start recursion from the root node (node 0)
    return rules

def interpret_condition_for_concept(feature_idx, operator, value, feature_names, original_categorical_features_list):
    # This function translates numerical conditions involving one-hot encoded features
    # back into more readable categorical statements.

    feature = feature_names[feature_idx]

    is_one_hot_feature = False
    original_feature_name = None
    category_value = None

    # Determine if it's a one-hot encoded feature
    for orig_cat_feat in original_categorical_features_list:
        if feature.startswith(f"{orig_cat_feat}_"):
            is_one_hot_feature = True
            original_feature_name = orig_cat_feat
            category_value = feature[len(f"{orig_cat_feat}_"):]
            break

    if is_one_hot_feature:
        # For one-hot encoded features, a value > 0.5 typically means the category is present (True)
        # and value <= 0.5 means the category is absent (False).
        if operator == '>' and value > 0.5:
             return f"**{original_feature_name}** IS '{category_value.replace('_', ' ')}'" # Replace underscore in category name for readability
        elif operator == '<=' and value < 0.5:
             return f"**{original_feature_name}** is NOT '{category_value.replace('_', ' ')}'"
        else:
            # This handles cases where a split might occur at 0.5 for a true binary feature
            # or if the tree oddly splits on a one-hot feature.
            return f"**{feature}** {operator} {value:.2f}"
    else:
        # For numerical features
        if operator == '<=':
            return f"**{feature}** is less than or equal to {value:.2f}"
        elif operator == '>':
            return f"**{feature}** is greater than {value:.2f}"
        return f"**{feature}** {operator} {value:.2f}" # Fallback, should not happen

def generate_actionable_concepts(rules, feature_names, original_categorical_features_list):
    concepts = []
    for i, rule in enumerate(rules):
        translated_conditions = []
        for feature_idx, operator, value in rule['conditions']:
            translated_conditions.append(interpret_condition_for_concept(feature_idx, operator, value, feature_names, original_categorical_features_list))

        # Basic concept string
        concept_str = f"If " + " AND ".join(translated_conditions) + ", then it is a **HIGH LIKELIHOOD of Overpayment**."

        # Add context about samples and purity
        concept_str += f" (Covers {rule['overpayment_samples']} 'Overpayment' cases out of {rule['total_samples_at_leaf']} total cases at this rule's leaf; Purity: {rule['purity']:.1%})"

        concepts.append({
            'concept_text': concept_str,
            'overpayment_samples': rule['overpayment_samples'],
            'total_samples_at_leaf': rule['total_samples_at_leaf'],
            'purity': rule['purity'],
            'conditions_raw': rule['conditions'] # Keep original raw conditions for SQL generation
        })
    # Prioritize concepts by the number of overpayment samples they cover
    return sorted(concepts, key=lambda x: x['overpayment_samples'], reverse=True)

# Get rules from the best trained decision tree model
overpayment_rules = get_tree_rules(best_decision_tree_model, all_feature_names, ['No Overpayment', 'Overpayment'])

# Generate and print actionable concepts
actionable_concepts = generate_actionable_concepts(overpayment_rules, all_feature_names, list(categorical_features))

if actionable_concepts:
    print("\nHere are the top actionable concepts/rules identified by the Decision Tree:")
    for i, concept_data in enumerate(actionable_concepts):
        print(f"\n--- Concept {i+1} ---")
        print(concept_data['concept_text'])
        print("\n**Corresponding SQL Query Pattern:**")

        # Generate SQL for each concept based on raw conditions
        sql_conditions = []
        for feature_idx, operator, value in concept_data['conditions_raw']:
            feature_name_in_sql = all_feature_names[feature_idx]

            is_one_hot_feature = False
            original_feature_name = None
            category_value = None

            for orig_cat_feat in categorical_features:
                if feature_name_in_sql.startswith(f"{orig_cat_feat}_"):
                    is_one_hot_feature = True
                    original_feature_name = orig_cat_feat
                    category_value = feature_name_in_sql[len(f"{orig_cat_feat}_"):]
                    break

            if is_one_hot_feature:
                if operator == '>' and value > 0.5: # Is this category
                    sql_conditions.append(f"{original_feature_name} = '{category_value.replace('_', ' ')}'")
                elif operator == '<=' and value < 0.5: # Is NOT this category
                    sql_conditions.append(f"{original_feature_name} != '{category_value.replace('_', ' ')}'")
                else:
                    # This case should ideally not be hit for clear categorical checks
                    sql_conditions.append(f"({feature_name_in_sql} {operator} {value:.2f})")
            else: # Numerical feature
                sql_conditions.append(f"{feature_name_in_sql} {operator} {value:.2f}")

        print("```sql")
        print("SELECT *")
        print("FROM Your_DRG_Medicare_Table")
        print("WHERE " + "\n  AND ".join(sql_conditions) + ";")
        print("```")
else:
    print("\nNo strong 'Overpayment' concepts could be extracted from the tree with the current purity and sample thresholds.")
    print("Consider adjusting `purity >= 0.7` or `total_samples_at_leaf >= 5` in `get_tree_rules` or `max_depth` in `param_grid` if you expect more rules.")

print("\n" + "="*50)
print("\n**How to Use These Actionable Concepts:**")
print("1.  **Direct Auditing**: Use the generated SQL queries to pull specific cases from your database for immediate review by auditors.")
print("2.  **Rule Development**: These concepts can be formalized into automated flagging rules in your existing fraud/abuse detection systems.")
print("3.  **Policy Review**: Patterns revealed might indicate gaps or ambiguities in current billing/reimbursement policies that need to be addressed.")
print("4.  **Provider Education**: If patterns consistently point to specific providers or types of providers, targeted education or intervention programs can be designed.")
print("5.  **Risk Scoring**: Integrate these rules into a risk scoring model, where cases matching multiple overpayment concepts receive a higher risk score.")
print("\nRemember to combine these automated concepts with your profound subject matter expertise for the most effective outcome.")
print("\n" + "="*50)

