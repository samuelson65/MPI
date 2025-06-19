import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
import re # For parsing Decision Tree rules
from sklearn.tree import _tree # <<<<<<<<<<<<<<<<<<<< CORRECTION HERE: Import _tree

# --- 1. Sample Data ---
data = {
    "claim_id": ["C1001", "C1002", "C1003", "C1004", "C1005", "C1006", "C1007", "C1008", "C1009", "C1010", "C1011", "C1012"],
    "provider_id": ["PRV001", "PRV001", "PRV002", "PRV001", "PRV003", "PRV002", "PRV001", "PRV003", "PRV002", "PRV001", "PRV004", "PRV001"],
    "drg_code": ["DRG-100", "DRG-100", "DRG-200", "DRG-100", "DRG-300", "DRG-200", "DRG-100", "DRG-300", "DRG-200", "DRG-100", "DRG-100", "DRG-200"],
    "comorbidities": ["E11.9,I10", "", "J45.9", "I50.9,N18.9", "K20.0", "", "E11.9", "I10", "J45.9,K20.0", "I50.9", "E11.9", "J45.9"],
    "discharge_status": ["Home", "Home", "SNF", "Home", "Home", "Home", "Home", "SNF", "Home", "Home", "Home", "Home"],
    "los": [15, 5, 7, 30, 8, 6, 12, 25, 9, 28, 4, 10], # Added C1011 (LOS 4, likely overpayment), C1012 (LOS 10, normal)
    "principal_diagnosis": ["I10", "I10", "J18.9", "I10", "K20.0", "J18.9", "I10", "K20.0", "J18.9", "I10", "I10", "J18.9"],
    "procedures": ["Z98.89", "Z98.89", "", "Z98.89,5A1945Z", "", "", "Z98.89", "", "", "Z98.89", "Z98.89", ""],
    "is_overpayment": [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0] # Updated with new overpayment for C1011
}
df = pd.DataFrame(data)

print("Original Data Head:")
print(df.head())
print("-" * 50)

# --- 2. Feature Engineering for Rule Mining & Decision Tree ---

# Discretize Length_of_Stay (LOS)
df['los_category'] = pd.cut(df['los'],
                            bins=[0, 7, 10, 20, np.inf], # Adjusted bins to create more distinct categories
                            labels=['Very_Short', 'Short', 'Medium', 'Long'],
                            right=False,
                            include_lowest=True) # Include lowest boundary

# Binarize all original features for Association Rule Mining (ARM)
df_arm_base = df[['claim_id']].copy()

# DRG Code
for drg in df['drg_code'].unique():
    df_arm_base[f'DRG_{drg}'] = (df['drg_code'] == drg).astype(int)

# Discharge Status
for status in df['discharge_status'].unique():
    df_arm_base[f'DISCHARGE_{status.replace(" ", "_")}'] = (df['discharge_status'] == status).astype(int)

# LOS Category
for los_cat in df['los_category'].unique():
    df_arm_base[f'LOS_{los_cat}'] = (df['los_category'] == los_cat).astype(int)

# Principal Diagnosis
for diag in df['principal_diagnosis'].unique():
    df_arm_base[f'P_DIAG_{diag}'] = (df['principal_diagnosis'] == diag).astype(int)

# Comorbidities (Multi-hot encoding)
mlb_comorb = MultiLabelBinarizer()
df['comorbidities_list'] = df['comorbidities'].apply(lambda x: [d.strip() for d in x.split(',') if d.strip()])
comorb_df = pd.DataFrame(mlb_comorb.fit_transform(df['comorbidities_list']),
                         columns=[f'C_DIAG_{c}' for c in mlb_comorb.classes_],
                         index=df.index)
df_arm_base = pd.concat([df_arm_base, comorb_df], axis=1)

# Procedures (Multi-hot encoding)
mlb_proc = MultiLabelBinarizer()
df['procedures_list'] = df['procedures'].apply(lambda x: [p.strip() for p in x.split(',') if p.strip()])
proc_df = pd.DataFrame(mlb_proc.fit_transform(df['procedures_list']),
                       columns=[f'PROC_{p}' for p in mlb_proc.classes_],
                       index=df.index)
df_arm_base = pd.concat([df_arm_base, proc_df], axis=1)

# Add overpayment label directly to ARM dataframe for later rule mining with target
df_arm_base['IS_OVERPAYMENT'] = df['is_overpayment'].astype(int)

# Drop claim_id for ARM processing
df_arm_features = df_arm_base.drop(columns=['claim_id'])

print("Transformed Data for Rule Mining (Head):")
print(df_arm_features.head())
print("-" * 50)

# --- 3. Association Rule Mining (Apriori) ---

# Find frequent itemsets
# Adjust min_support as needed. Smaller datasets might need lower support.
frequent_itemsets = apriori(df_arm_features, min_support=0.15, use_colnames=True) # Slightly lower support for more combos
print(f"Found {len(frequent_itemsets)} frequent itemsets.")

# Generate association rules
# Filter for rules that strongly predict IS_OVERPAYMENT
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6) # High confidence
rules = rules[rules['lift'] > 1.2] # Lift greater than 1.2 (strong positive correlation)
rules = rules.sort_values(by=["confidence", "lift"], ascending=False).reset_index(drop=True)

print(f"Found {len(rules)} association rules (filtered for high confidence/lift):")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
print("-" * 50)

# --- 4. Identify "Best Combos" (Candidate Concepts) from ARM ---
# We're looking for antecedents that are strong predictors of IS_OVERPAYMENT
overpayment_predictive_rules = rules[rules['consequents'].apply(lambda x: 'IS_OVERPAYMENT' in x)]

best_combos = []
for index, row in overpayment_predictive_rules.iterrows():
    # Convert frozenset to a sorted tuple for consistency
    combo_tuple = tuple(sorted(list(row['antecedents'])))
    best_combos.append({
        'combo': combo_tuple,
        'confidence': row['confidence'],
        'lift': row['lift']
    })

# Sort combos by confidence, then lift
best_combos.sort(key=lambda x: (x['confidence'], x['lift']), reverse=True)

print(f"\nIdentified {len(best_combos)} 'Best Combos' from ARM strongly predicting Overpayment:")
for bc in best_combos[:5]: # Print top 5
    print(f"- Antecedents: {', '.join(bc['combo'])} | Conf: {bc['confidence']:.2f}, Lift: {bc['lift']:.2f}")
print("-" * 50)

# --- 5. Enhance Decision Tree Input with "Best Combo" Features ---

# Create a new DataFrame for Decision Tree features, starting with original binarized features
# Exclude 'IS_OVERPAYMENT' as it's the target
X_dt = df_arm_base.drop(columns=['claim_id', 'IS_OVERPAYMENT'])

# Add new columns for each 'best combo' identified by ARM
arm_combo_feature_names = []
for i, bc in enumerate(best_combos):
    # Create a cleaner name for the combo feature
    combo_items_short = [item.replace('DRG_', '').replace('LOS_', '').replace('P_DIAG_', '').replace('C_DIAG_', '').replace('PROC_', '') for item in bc['combo']]
    combo_name = f"ARM_COMBO_{'_'.join(combo_items_short)}" # Generate a descriptive name
    
    # Ensure name is unique if same items appear in different combos
    # This simple approach prefixes with index if name already exists
    original_combo_name = combo_name
    counter = 1
    while combo_name in X_dt.columns:
        combo_name = f"{original_combo_name}_{counter}"
        counter += 1
        
    X_dt[combo_name] = 0 # Initialize column
    arm_combo_feature_names.append(combo_name)

    # For each claim, check if it matches the 'antecedents' of this best combo
    for idx, row in df_arm_base.iterrows():
        match = True
        for item in bc['combo']:
            if item not in row.index or row[item] == 0: # Check if the item exists and is 1
                match = False
                break
        if match:
            X_dt.loc[idx, combo_name] = 1

print("\nDecision Tree Features with Added ARM Combo Features (Head):")
print(X_dt.head())
print("-" * 50)

# --- 6. Decision Tree for Building Conditional Queries (Concepts) ---

# Prepare target variable
y_dt = df['is_overpayment']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=0.3, random_state=42, stratify=y_dt)

# Train a Decision Tree Classifier
# Max depth limited for interpretability. Class_weight for imbalanced data.
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
dt_classifier.fit(X_train, y_train)

print(f"Decision Tree Accuracy on Test Set: {dt_classifier.score(X_test, y_test):.2f}")
print("-" * 50)

# --- 7. Extract Conditional Queries (Concepts) from the Decision Tree ---

# Function to extract rules in a human-readable format
def get_dt_rules(tree, feature_names, class_names):
    tree_rules_text = export_text(tree, feature_names=list(feature_names), class_names=list(map(str, class_names)))
    return tree_rules_text

dt_rules_raw = get_dt_rules(dt_classifier, X_dt.columns, dt_classifier.classes_)
print("\nRaw Decision Tree Rules (for reference):")
print(dt_rules_raw)

print("\n--- Automatically Generated Concepts (Conditional Queries) for Overpayment Detection ---")
print("These are rules derived from the Decision Tree, incorporating 'Best Combos' from Rule Mining:\n")

# This parsing function is more robust for nested conditions.
# It reconstructs paths to 'class: 1' (overpayment) leaves.
def parse_dt_rules_to_concepts(tree, feature_names, class_names, target_class_label='1'):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    overpayment_concepts = []

    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            # Left child (True branch)
            recurse(tree_.children_left[node], path + [(name, '<=', threshold)]) # <= threshold means feature value is 0 or less
            # Right child (False branch)
            recurse(tree_.children_right[node], path + [(name, '>', threshold)]) # > threshold means feature value is 1 or more
        else: # Leaf node
            # Check if this leaf node represents the target class (overpayment)
            # tree_.value[node] gives [num_samples_class0, num_samples_class1]
            if np.argmax(tree_.value[node]) == class_names.index(int(target_class_label)): 
                
                concept_conditions = []
                for cond_feature, operator, value in path:
                    # Special handling for binary features (which are the majority after binarization)
                    # A condition like 'FEATURE <= 0.5' means FEATURE is 0 (False/Absent)
                    # A condition like 'FEATURE > 0.5' means FEATURE is 1 (True/Present)
                    if operator == '<=' and value == 0.5:
                         concept_conditions.append(f"NOT ({cond_feature.replace('_', ' ').strip()})")
                    elif operator == '>' and value == 0.5:
                        concept_conditions.append(f"{cond_feature.replace('_', ' ').strip()}")
                    else: # Fallback for non-binary (though less expected here)
                        concept_conditions.append(f"{cond_feature} {operator} {value:.2f}")

                concept = " AND ".join(concept_conditions)
                if concept:
                    overpayment_concepts.append(f"CONCEPT: IF ({concept}) THEN (POTENTIAL OVERPAYMENT)")

    recurse(0, [])
    
    # Remove duplicates if any (e.g., from multiple paths leading to same leaf)
    return sorted(list(set(overpayment_concepts)))

generated_concepts = parse_dt_rules_to_concepts(dt_classifier, X_dt.columns, dt_classifier.classes_, target_class_label='1')

for concept in generated_concepts:
    print(concept)

# --- Optional: Visualize the Decision Tree ---
# This requires graphviz to be installed (pip install graphviz)
# from sklearn.tree import plot_tree
# plt.figure(figsize=(30, 15))
# plot_tree(dt_classifier, feature_names=X_dt.columns, class_names=['Not Overpayment', 'Overpayment'],
#           filled=True, rounded=True, fontsize=10, proportion=False)
# plt.title("Decision Tree for Overpayment Detection (with ARM Combo Features)")
# plt.show()

print("\nScript finished.")
