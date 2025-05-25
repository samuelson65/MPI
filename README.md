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
np.random.seed(42)

num_samples = 1000

discharge_statuses = ['Discharged Home', 'Transferred to SNF', 'Died', 'Discharged Against Medical Advice', 'Transferred to Acute Care']
is_present_options = ['Yes', 'No']

data = {
    'Length_of_Stay': np.random.randint(1, 60, num_samples),
    'Discharge_Status': np.random.choice(discharge_statuses, num_samples, p=[0.6, 0.2, 0.05, 0.05, 0.1]),
    'Procedure_Count': np.random.randint(0, 10, num_samples),
    'Diff_Between_Proc_Performed_Date': np.random.randint(0, 30, num_samples),
    'MCC_Count': np.random.randint(0, 5, num_samples),
    'CC_Count': np.random.randint(0, 8, num_samples),
    'Is_Catheter_Present': np.random.choice(is_present_options, num_samples, p=[0.2, 0.8]),
    'Is_Stent_Present': np.random.choice(is_present_options, num_samples, p=[0.15, 0.85]),
}
df = pd.DataFrame(data)

df['Overpayment'] = 0
df.loc[(df['Length_of_Stay'] > 45) & (df['Procedure_Count'] < 2) & (df['Discharge_Status'] == 'Transferred to SNF'), 'Overpayment'] = 1
df.loc[(df['Length_of_Stay'] < 5) & (df['MCC_Count'] >= 3) & (df['CC_Count'] >= 5), 'Overpayment'] = 1
df.loc[((df['Is_Catheter_Present'] == 'Yes') | (df['Is_Stent_Present'] == 'Yes')) & (df['Diff_Between_Proc_Performed_Date'] > 20), 'Overpayment'] = 1
df.loc[(df['Procedure_Count'] == 0) & (df['MCC_Count'] >= 4) & (df['CC_Count'] >= 6), 'Overpayment'] = 1

df.loc[df['Overpayment'] == 0, 'Overpayment'] = np.random.choice([0, 1], sum(df['Overpayment'] == 0), p=[0.97, 0.03])

if df['Overpayment'].sum() < 30:
    missing_overpayments = 30 - df['Overpayment'].sum()
    if missing_overpayments > 0:
        overpayment_indices = np.random.choice(df.index[df['Overpayment'] == 0], missing_overpayments, replace=False)
        df.loc[overpayment_indices, 'Overpayment'] = 1

print(f"Generated dummy data with {len(df)} samples. Overpayment cases: {df['Overpayment'].sum()}")
print(df.head())

X = df.drop('Overpayment', axis=1)
y = df['Overpayment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")
print(f"Overpayment in training set: {y_train.sum()} ({y_train.mean():.2%})")
print(f"Overpayment in test set: {y_test.sum()} ({y_test.mean():.2%})")


# --- 2. Identify Categorical and Numerical Features ---
categorical_features = ['Discharge_Status', 'Is_Catheter_Present', 'Is_Stent_Present']
numerical_features = ['Length_of_Stay', 'Procedure_Count', 'Diff_Between_Proc_Performed_Date', 'MCC_Count', 'CC_Count']

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

param_grid = {
    'classifier__max_depth': [7, 10, 15, 20],
    'classifier__min_samples_split': [5, 10, 20],
    'classifier__min_samples_leaf': [3, 5, 10],
    'classifier__max_features': [None, 'sqrt', 'log2'],
    'classifier__class_weight': [None, 'balanced']
}

print("\nStarting Grid Search for best Decision Tree hyperparameters with 5-fold Cross-Validation...")
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)
print("Grid Search complete.")

best_model_pipeline = grid_search.best_estimator_
best_decision_tree_model = best_model_pipeline.named_steps['classifier']

print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Best ROC AUC score (on training folds): {grid_search.best_score_:.4f}")

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
def get_tree_rules(tree_model, feature_names, class_names):
    tree_ = tree_model.tree_
    rules = []
    
    # Assuming 'Overpayment' is class 1
    overpayment_class_id = class_names.index('Overpayment') if 'Overpayment' in class_names else 1 # Default to 1

    def recurse(node, path_conditions):
        if tree_.feature[node] != -2: # Not a leaf node
            feature_idx = tree_.feature[node]
            threshold = tree_.threshold[node]

            # Left child (True condition: feature <= threshold)
            recurse(tree_.children_left[node], path_conditions + [(feature_idx, '<=', threshold)])

            # Right child (False condition: feature > threshold)
            recurse(tree_.children_right[node], path_conditions + [(feature_idx, '>', threshold)])
        else: # Leaf node
            total_samples_at_leaf = tree_.n_node_samples[node]
            
            # Safely get the count for the 'Overpayment' class
            # tree_.value[node] is typically [[count_class_0, count_class_1, ...]]
            # We want the count for the 'Overpayment' class (class_id 1 by default)
            num_overpayment_samples = tree_.value[node][0, overpayment_class_id]

            # Ensure total_samples_at_leaf is not zero to avoid division by zero
            # And ensure num_overpayment_samples doesn't exceed total_samples_at_leaf (should not happen if correctly extracted)
            if total_samples_at_leaf > 0:
                purity = num_overpayment_samples / total_samples_at_leaf
            else:
                purity = 0.0 # No samples, so 0 purity for overpayment
            
            # Only consider rules if the leaf strongly predicts 'Overpayment' and covers enough samples
            if num_overpayment_samples > 0 and total_samples_at_leaf >= 5 and purity >= 0.7:
                rules.append({
                    'conditions': path_conditions,
                    'predicted_class': class_names[overpayment_class_id],
                    'overpayment_samples': int(num_overpayment_samples),
                    'total_samples_at_leaf': int(total_samples_at_leaf),
                    'purity': purity
                })

    recurse(0, [])
    return rules

def interpret_condition_for_concept(feature_idx, operator, value, feature_names, original_categorical_features_list):
    feature = feature_names[feature_idx]

    is_one_hot_feature = False
    original_feature_name = None
    category_value = None

    for orig_cat_feat in original_categorical_features_list:
        if feature.startswith(f"{orig_cat_feat}_"):
            is_one_hot_feature = True
            original_feature_name = orig_cat_feat
            category_value = feature[len(f"{orig_cat_feat}_"):].replace('_', ' ')
            break

    if is_one_hot_feature:
        if operator == '>' and value > 0.5:
             return f"**{original_feature_name}** IS '{category_value}'"
        elif operator == '<=' and value < 0.5:
             return f"**{original_feature_name}** is NOT '{category_value}'"
        else:
            return f"**{feature}** {operator} {value:.2f}"
    else:
        if operator == '<=':
            return f"**{feature}** is less than or equal to {value:.2f}"
        elif operator == '>':
            return f"**{feature}** is greater than {value:.2f}"
        return f"**{feature}** {operator} {value:.2f}"

def generate_actionable_concepts(rules, feature_names, original_categorical_features_list):
    concepts = []
    for i, rule in enumerate(rules):
        translated_conditions = []
        for feature_idx, operator, value in rule['conditions']:
            translated_conditions.append(interpret_condition_for_concept(feature_idx, operator, value, feature_names, original_categorical_features_list))

        concept_str = f"If " + " AND ".join(translated_conditions) + ", then it is a **HIGH LIKELIHOOD of Overpayment**."

        concept_str += f" (Covers {rule['overpayment_samples']} 'Overpayment' cases out of {rule['total_samples_at_leaf']} total cases at this rule's leaf; Purity: {rule['purity']:.1%})"

        concepts.append({
            'concept_text': concept_str,
            'overpayment_samples': rule['overpayment_samples'],
            'total_samples_at_leaf': rule['total_samples_at_leaf'],
            'purity': rule['purity'],
            'conditions_raw': rule['conditions']
        })
    return sorted(concepts, key=lambda x: x['overpayment_samples'], reverse=True)

overpayment_rules = get_tree_rules(best_decision_tree_model, all_feature_names, ['No Overpayment', 'Overpayment'])

actionable_concepts = generate_actionable_concepts(overpayment_rules, all_feature_names, list(categorical_features))

if actionable_concepts:
    print("\nHere are the top actionable concepts/rules identified by the Decision Tree:")
    for i, concept_data in enumerate(actionable_concepts):
        print(f"\n--- Concept {i+1} ---")
        print(concept_data['concept_text'])
        print("\n**Corresponding SQL Query Pattern:**")

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
                    category_value = feature_name_in_sql[len(f"{orig_cat_feat}_"):].replace('_', ' ')
                    break

            if is_one_hot_feature:
                if operator == '>' and value > 0.5:
                    sql_conditions.append(f"{original_feature_name} = '{category_value}'")
                elif operator == '<=' and value < 0.5:
                    sql_conditions.append(f"{original_feature_name} != '{category_value}'")
                else:
                    sql_conditions.append(f"({feature_name_in_sql} {operator} {value:.2f})")
            else:
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

