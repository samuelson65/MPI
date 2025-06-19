import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer # For handling multiple diagnosis codes
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Sample Data (Similar to your described dataset) ---
# Added 'is_overpayment' for demonstration, which would come from your CatBoost flags or manual labels.
data = {
    "claim_id": ["C1001", "C1002", "C1003", "C1004", "C1005", "C1006", "C1007", "C1008", "C1009", "C1010"],
    "provider_id": ["PRV001", "PRV001", "PRV002", "PRV001", "PRV003", "PRV002", "PRV001", "PRV003", "PRV002", "PRV001"],
    "drg_code": ["DRG-100", "DRG-100", "DRG-200", "DRG-100", "DRG-300", "DRG-200", "DRG-100", "DRG-300", "DRG-200", "DRG-100"],
    "comorbidities": ["E11.9,I10", "", "J45.9", "I50.9,N18.9", "K20.0", "", "E11.9", "I10", "J45.9,K20.0", "I50.9"],
    "discharge_status": ["Home", "Home", "SNF", "Home", "Home", "Home", "Home", "SNF", "Home", "Home"],
    "los": [15, 5, 7, 30, 8, 6, 12, 25, 9, 28],
    "principal_diagnosis": ["I10", "I10", "J18.9", "I10", "K20.0", "J18.9", "I10", "K20.0", "J18.9", "I10"],
    "procedures": ["Z98.89", "Z98.89", "", "Z98.89,5A1945Z", "", "", "Z98.89", "", "", "Z98.89"],
    # This is your 'target' variable: 1 for overpayment, 0 for not.
    # From CatBoost or validated manual audits.
    "is_overpayment": [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

print("Original Data Head:")
print(df.head())
print("-" * 50)

# --- 2. Feature Engineering for Rule Mining & Decision Tree ---

# Discretize Length_of_Stay (LOS)
# This is crucial for rule mining and can make DT rules more interpretable
df['los_category'] = pd.cut(df['los'],
                            bins=[0, 7, 14, 21, np.inf],
                            labels=['Short', 'Medium', 'Long', 'VeryLong'],
                            right=False)

# Binarize categorical features for Association Rule Mining (ARM)
# Each unique category/item becomes a boolean column
df_arm = df[['claim_id']].copy() # Start with claim_id for transaction context

# DRG Code
for drg in df['drg_code'].unique():
    df_arm[f'DRG_{drg}'] = (df['drg_code'] == drg).astype(int)

# Discharge Status
for status in df['discharge_status'].unique():
    df_arm[f'DISCHARGE_{status.replace(" ", "_")}'] = (df['discharge_status'] == status).astype(int)

# LOS Category
for los_cat in df['los_category'].unique():
    df_arm[f'LOS_{los_cat}'] = (df['los_category'] == los_cat).astype(int)

# Principal Diagnosis
for diag in df['principal_diagnosis'].unique():
    df_arm[f'P_DIAG_{diag}'] = (df['principal_diagnosis'] == diag).astype(int)

# Comorbidities (Multi-hot encoding for multiple diagnoses per claim)
mlb_comorb = MultiLabelBinarizer()
df['comorbidities_list'] = df['comorbidities'].apply(lambda x: [d.strip() for d in x.split(',') if d.strip()])
comorb_df = pd.DataFrame(mlb_comorb.fit_transform(df['comorbidities_list']),
                         columns=[f'C_DIAG_{c}' for c in mlb_comorb.classes_],
                         index=df.index)
df_arm = pd.concat([df_arm, comorb_df], axis=1)

# Procedures (Multi-hot encoding for multiple procedures per claim)
mlb_proc = MultiLabelBinarizer()
df['procedures_list'] = df['procedures'].apply(lambda x: [p.strip() for p in x.split(',') if p.strip()])
proc_df = pd.DataFrame(mlb_proc.fit_transform(df['procedures_list']),
                       columns=[f'PROC_{p}' for p in mlb_proc.classes_],
                       index=df.index)
df_arm = pd.concat([df_arm, proc_df], axis=1)

# Add overpayment label directly to ARM dataframe for later rule mining with target
df_arm['IS_OVERPAYMENT'] = df['is_overpayment'].astype(int)

# Drop claim_id as it's not a feature for ARM, but keep for context if needed later
df_arm_features = df_arm.drop(columns=['claim_id'])

print("Transformed Data for Rule Mining (Head):")
print(df_arm_features.head())
print("-" * 50)

# --- 3. Association Rule Mining (Apriori) ---

# Find frequent itemsets
# min_support: Minimum frequency for an itemset to be considered 'frequent'.
# Adjust this based on your dataset size and desired granularity.
frequent_itemsets = apriori(df_arm_features, min_support=0.2, use_colnames=True)
print(f"Found {len(frequent_itemsets)} frequent itemsets:")
print(frequent_itemsets.head())
print("-" * 50)

# Generate association rules
# min_confidence: Minimum confidence for a rule (e.g., if A, then B, with X% confidence)
# lift: A measure of how much more likely A and B are to occur together than independently. >1 indicates positive correlation.
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.1)
rules = rules.sort_values(by="lift", ascending=False).reset_index(drop=True)

print(f"Found {len(rules)} association rules (sorted by Lift):")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
print("-" * 50)

# --- 4. Identify Rules related to Overpayment ---
# These are strong candidates for new concepts
overpayment_rules = rules[rules['consequents'].apply(lambda x: 'IS_OVERPAYMENT' in x)]
print("\nAssociation Rules where 'IS_OVERPAYMENT' is the consequent:")
if not overpayment_rules.empty:
    for index, row in overpayment_rules.iterrows():
        antecedents = ", ".join(list(row['antecedents']))
        print(f"IF ({antecedents}) THEN (IS_OVERPAYMENT) -- Support: {row['support']:.2f}, Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f}")
else:
    print("No direct 'IS_OVERPAYMENT' rules found with current thresholds. Adjust min_support/min_confidence.")
print("-" * 50)


# --- 5. Decision Tree for Building Conditional Queries (Concepts) ---

# Prepare data for Decision Tree
# We'll use the binarized features created for ARM, but drop the 'is_overpayment' target for now
X = df_arm_features.drop(columns=['IS_OVERPAYMENT'])
y = df['is_overpayment'] # Original target column

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # stratify helps with imbalanced classes

# Train a Decision Tree Classifier
# Max depth can be limited to make rules more interpretable
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced') # 'balanced' helps with imbalanced target
dt_classifier.fit(X_train, y_train)

print(f"Decision Tree Accuracy on Test Set: {dt_classifier.score(X_test, y_test):.2f}")
print("-" * 50)

# --- 6. Extract Conditional Queries (Concepts) from the Decision Tree ---

# Function to extract rules in a human-readable format
def get_dt_rules(tree, feature_names, class_names):
    tree_rules = export_text(tree, feature_names=list(feature_names), class_names=list(map(str, class_names)))
    return tree_rules

print("\nConditional Queries (Concepts) Extracted from Decision Tree:")
dt_rules_text = get_dt_rules(dt_classifier, X.columns, dt_classifier.classes_)
print(dt_rules_text)
print("-" * 50)

# Optional: Visualize the Decision Tree
# This can be helpful for understanding the generated rules graphically
# plt.figure(figsize=(20,10))
# plot_tree(dt_classifier, feature_names=X.columns, class_names=['Not Overpayment', 'Overpayment'], filled=True, rounded=True, fontsize=10)
# plt.title("Decision Tree for Overpayment Detection")
# plt.show()


# --- Interpreting and Generating Actionable Concepts ---

print("\n--- Actionable Concepts (Conditional Queries) for Overpayment Detection ---")
print("These are rules derived automatically from the data:\n")

# Parse the Decision Tree text output to formulate concepts
# This is a basic parsing; more robust parsing might be needed for complex trees
for line in dt_rules_text.split('\n'):
    if '|--- class: 1' in line: # Focus on paths leading to overpayment (class 1)
        # Extract the conditions leading to this leaf
        conditions = line.strip().split('|---')[0].strip() # Get the indentation part
        path = []
        for l in dt_rules_text.split('\n'):
            if l.startswith(conditions) and '|--- class: 1' not in l:
                path.append(l.strip())
            elif l.startswith(conditions) and '|--- class: 1' in l:
                 path.append(l.strip().split('|---')[0].strip()) # Add the condition part of the leaf node
                 break

        # Reconstruct the rule
        rule_conditions = []
        current_indent = -1
        for p_line in path:
            indent = p_line.count('|   ')
            if indent > current_indent:
                # This is a new condition for the current path
                condition_text = p_line.replace('|   ', '').strip()
                if 'class: ' not in condition_text: # Avoid adding class label as a condition
                    rule_conditions.append(condition_text)
            else: # Go up the tree or sideways
                while indent <= current_indent:
                    rule_conditions.pop()
                    current_indent -= 1
                condition_text = p_line.replace('|   ', '').strip()
                if 'class: ' not in condition_text:
                    rule_conditions.append(condition_text)
            current_indent = indent
            
        final_rule = " AND ".join([c for c in rule_conditions if c]) # Remove empty strings from conditions
        if final_rule: # Ensure there's a rule
            print(f"CONCEPT: IF ({final_rule}) THEN (POTENTIAL OVERPAYMENT)")

# Additional insights from Association Rules
print("\n--- Additional Insights from Association Rule Mining (Potential Concepts) ---")
print("These are frequent co-occurrences that may not be direct prediction rules, but indicate patterns:")
# Filter rules to show interesting co-occurrences, e.g., high support/confidence combinations
for index, row in rules.head(5).iterrows(): # Show top 5 rules by lift
    antecedents = ", ".join(list(row['antecedents']))
    consequents = ", ".join(list(row['consequents']))
    print(f"PATTERN: IF ({antecedents}) THEN ({consequents}) -- Support: {row['support']:.2f}, Confidence: {row['confidence']:.2f}")

