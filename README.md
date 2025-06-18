import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# Libraries for Rule Extraction (need to be installed)
try:
    from skrules import SkopeRulesClassifier
except ImportError:
    print("SkopeRules not installed. Please run: pip install skope-rules")
    SkopeRulesClassifier = None

try:
    from rulefit import RuleFit
except ImportError:
    print("RuleFit not installed. Please run: pip install rulefit")
    RuleFit = None # Set to None if import fails


# --- 1. Create a Sample DataFrame with Medical Billing Features ---
np.random.seed(42)
num_samples = 2000 # Increased samples for better rule learning

# Realistic-ish codes and categories
drg_codes = ['291', '292', '313', '470', '871', '872', '207', '603'] # Common DRGs
diag_codes = ['I50.9', 'J18.9', 'I10', 'E11.9', 'G81.9', 'N18.9', 'K21.9'] # ICD-10 examples
proc_codes = ['0HZG0ZZ', '02HQ0ZZ', '0PT0XZZ', '0SR90ZZ', '0JB03ZZ', '0DW40ZZ'] # PCS examples
discharge_statuses = ['Discharged Home', 'Transferred to SNF', 'Died', 'Discharged AMA']
is_present_options = ['Yes', 'No']

data = {
    'DRG_Code': np.random.choice(drg_codes, num_samples, p=[0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15]),
    'Diag_Code': np.random.choice(diag_codes, num_samples, p=[0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1]),
    'Proc_Code': np.random.choice(proc_codes, num_samples, p=[0.2, 0.2, 0.15, 0.15, 0.1, 0.2]),
    'Length_of_Stay': np.random.randint(1, 45, num_samples), # Days
    'Total_Charges': np.random.randint(5000, 150000, num_samples), # Dollars
    'MCC_Count': np.random.randint(0, 5, num_samples), # Major Complication/Comorbidity Count
    'CC_Count': np.random.randint(0, 8, num_samples), # Complication/Comorbidity Count
    'Is_Catheter_Present': np.random.choice(is_present_options, num_samples, p=[0.25, 0.75]),
    'Is_Stent_Present': np.random.choice(is_present_options, num_samples, p=[0.15, 0.85]),
    'Discharge_Status': np.random.choice(discharge_statuses, num_samples, p=[0.7, 0.15, 0.1, 0.05]),
}
df = pd.DataFrame(data)

# Create a target variable 'Overpayment' based on some hypothetical complex rules
# These rules are designed to make some cases 'Overpayment'
df['Overpayment'] = 0

# Rule 1: High charges for short stay with specific DRG/Diagnosis
df.loc[(df['DRG_Code'] == '871') & (df['Length_of_Stay'] < 5) & (df['Total_Charges'] > 80000), 'Overpayment'] = 1
df.loc[(df['Diag_Code'] == 'I10') & (df['Length_of_Stay'] < 3) & (df['Total_Charges'] > 50000), 'Overpayment'] = 1

# Rule 2: High MCC/CC count for common DRG with low procedure count
df.loc[(df['DRG_Code'].isin(['291', '292'])) & (df['MCC_Count'] >= 3) & (df['CC_Count'] >= 5) & (df['Proc_Code'] == '0HZG0ZZ'), 'Overpayment'] = 1

# Rule 3: Presence of catheter/stent with unusually long stay and average charges
df.loc[((df['Is_Catheter_Present'] == 'Yes') | (df['Is_Stent_Present'] == 'Yes')) &
       (df['Length_of_Stay'] > 30) & (df['Total_Charges'].between(20000, 60000)), 'Overpayment'] = 1

# Rule 4: Transferred to SNF with very low Length_of_Stay and high charges (might indicate short but complex, potentially overcoded)
df.loc[(df['Discharge_Status'] == 'Transferred to SNF') & (df['Length_of_Stay'] < 7) & (df['Total_Charges'] > 70000), 'Overpayment'] = 1

# Introduce some random noise to make it less perfectly separable
# Keep the target class imbalanced (more 'No Overpayment')
df.loc[df['Overpayment'] == 0, 'Overpayment'] = np.random.choice([0, 1], sum(df['Overpayment'] == 0), p=[0.97, 0.03])

# Ensure at least some positive cases for proper training
if df['Overpayment'].sum() < 50:
    num_to_add = 50 - df['Overpayment'].sum()
    if num_to_add > 0:
        # Find indices of 'No Overpayment' to flip
        no_overpayment_indices = df.index[df['Overpayment'] == 0].tolist()
        flip_indices = np.random.choice(no_overpayment_indices, num_to_add, replace=False)
        df.loc[flip_indices, 'Overpayment'] = 1

print(f"Generated dummy data with {len(df)} samples. Overpayment cases: {df['Overpayment'].sum()}")
print(df.head())

# Separate features (X) and target (y)
X = df.drop('Overpayment', axis=1)
y = df['Overpayment']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")
print(f"Overpayment in training set: {y_train.sum()} ({y_train.mean():.2%})")
print(f"Overpayment in test set: {y_test.sum()} ({y_test.mean():.2%})")


# --- 2. Preprocessing: One-Hot Encode Categorical Features ---
categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough' # Keep numerical features as they are
)

# Fit and transform training data, transform test data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get feature names after one-hot encoding for the models
encoded_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = list(encoded_feature_names) + list(numerical_features)

# Convert processed data back to DataFrame for better compatibility with some libraries
# (SkopeRules/RuleFit can often take numpy arrays directly, but DataFrame helps with feature names)
X_train_processed_df = pd.DataFrame(X_train_processed, columns=all_feature_names, index=X_train.index)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=all_feature_names, index=X_test.index)

print(f"\nProcessed data shape (X_train_processed_df): {X_train_processed_df.shape}")
print(f"First 5 rows of processed data:\n{X_train_processed_df.head()}")


# --- 3. SkopeRules Example ---
if SkopeRulesClassifier:
    print("\n" + "="*50)
    print("--- SkopeRules Example ---")
    print("="*50)

    # Initialize SkopeRulesClassifier
    # precision_min, recall_min are crucial for controlling rule quality
    # n_estimators: number of base estimators (trees)
    # max_depth: maximum depth of individual trees
    sk_clf = SkopeRulesClassifier(
        n_estimators=100, # More estimators can find more rules
        max_depth=5,      # Keep individual trees shallow for simpler rules
        precision_min=0.6, # Minimum precision for extracted rules (e.g., 60% of cases flagged by this rule are truly overpayments)
        recall_min=0.01,  # Minimum recall (even if a rule only covers a small portion of overpayments, it can be useful if precise)
        feature_names=all_feature_names,
        random_state=42,
        n_jobs=-1,        # Use all available CPU cores
        verbose=0         # Suppress verbose output during training
    )

    print("Fitting SkopeRules Classifier...")
    sk_clf.fit(X_train_processed_df, y_train)
    print("SkopeRules fitting complete.")

    print(f"\nFound {len(sk_clf.rules_)} rules.")
    print("Top 10 SkopeRules for Overpayment (ranked by precision):")

    # Rules are stored as (rule_string, precision, recall) tuples
    # Sort rules by precision (descending) then by recall (descending)
    sorted_rules = sorted(sk_clf.rules_, key=lambda x: (x[1], x[2]), reverse=True)

    for i, (rule, precision, recall) in enumerate(sorted_rules[:10]):
        print(f"\nRule {i+1}:")
        print(f"  Conditions: {rule}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print("-" * 30)

    # Evaluate SkopeRules on the test set
    y_pred_sk = sk_clf.predict(X_test_processed_df)
    # SkopeRules doesn't directly provide predict_proba, but you can infer scores
    # based on how many rules fire or specific internal methods if needed.
    # For a simple classification report:
    print("\nSkopeRules Classification Report on Test Set:")
    print(classification_report(y_test, y_pred_sk, target_names=['No Overpayment', 'Overpayment']))

    # To get a single ROC AUC score (if applicable):
    # This might require custom score aggregation if rules are not directly probabilistic.
    # For simplicity, we'll skip ROC AUC unless a probabilistic output is easily available.


# --- 4. RuleFit Example ---
if RuleFit:
    print("\n" + "="*50)
    print("--- RuleFit Example ---")
    print("="*50)

    # Initialize RuleFit model
    # tree_generator: The base estimator to extract rules from (e.g., GradientBoostingClassifier)
    # rfmode='classify' for classification tasks.
    # max_rules: Maximum number of rules to include in the final model (after Lasso selection)
    # tree_generator needs to be an instance of a scikit-learn tree ensemble
    from sklearn.ensemble import GradientBoostingClassifier
    gb_classifier = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)

    rf_model = RuleFit(
        tree_generator=gb_classifier,
        rfmode='classify', # For classification problems
        max_rules=2000,   # Generates many rules, Lasso will select
        random_state=42,
        n_jobs=-1
    )

    print("Fitting RuleFit Model...")
    # RuleFit expects numpy arrays, which X_train_processed is.
    # It also needs feature_names to interpret the rules.
    rf_model.fit(X_train_processed_df.values, y_train.values, feature_names=all_feature_names)
    print("RuleFit fitting complete.")

    print("\nTop 15 RuleFit Rules and Linear Terms (ranked by importance):")
    # Get the learned rules and linear terms
    rules = rf_model.get_rules()

    # Filter for terms with non-zero coefficients (selected by Lasso)
    # and sort by absolute importance (magnitude of coefficient)
    rules_filtered = rules[rules.coef != 0].sort_values(by="importance", ascending=False)

    for i, row in rules_filtered.head(15).iterrows():
        if row['type'] == 'rule':
            # 'rule' column contains the string representation of the rule
            print(f"Rule {i+1}: IF {row['rule']}")
            print(f"  Coefficient: {row['coef']:.4f}, Importance: {row['importance']:.4f}")
            print(f"  Support: {row['support']:.4f} (proportion of samples where rule is true)")
        else: # type == 'linear' (original feature)
            print(f"Linear Term {i+1}: {row['feature']}")
            print(f"  Coefficient: {row['coef']:.4f}, Importance: {row['importance']:.4f}")
        print("-" * 30)

    # Evaluate RuleFit on the test set
    y_pred_rf = rf_model.predict(X_test_processed_df.values)
    y_pred_proba_rf = rf_model.predict_proba(X_test_processed_df.values) # RuleFit directly provides probabilities

    print("\nRuleFit Classification Report on Test Set:")
    print(classification_report(y_test, y_pred_rf, target_names=['No Overpayment', 'Overpayment']))
    print(f"RuleFit ROC AUC Score on Test Set: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")


# --- 5. Explanation of Output and Actionable Insights ---
print("\n" + "="*50)
print("--- How to Interpret and Use These Rules ---")
print("="*50)

print("\n**SkopeRules Output:**")
print("- You get a list of `IF-THEN` rules (conditions).")
print("- Each rule comes with its `Precision` and `Recall`.")
print("  - **Precision:** Of all cases flagged by this specific rule, what percentage were *actual* overpayments?")
print("  - **Recall:** Of all *actual* overpayments, what percentage were caught by this specific rule?")
print("- **Actionable Use:**")
print("  - **High Precision Rules:** These are excellent for direct auditing. An auditor can confidently review cases that trigger these rules, as a high percentage will likely be true positives.")
print("  - **Rule Prioritization:** Sort rules by precision, then recall, to identify your most reliable 'triggers'.")
print("  - **SQL Queries:** Translate these rules directly into SQL `WHERE` clauses to query your billing database.")
print("    Example: `SELECT * FROM claims WHERE DRG_Code = '871' AND Length_of_Stay < 5 AND Total_Charges > 80000;`")

print("\n**RuleFit Output:**")
print("- You get two types of terms: `Linear Terms` (original features) and `Rules` (combinations of features).")
print("- Each term has a `Coefficient` and `Importance`:")
print("  - **Coefficient:**")
print("    - For a **linear term**: A positive coefficient means higher values of that feature increase the likelihood of overpayment, and vice-versa. The magnitude indicates the strength.")
print("    - For a **rule**: A positive coefficient means that if the rule's conditions are met (the binary rule feature is 1), it increases the likelihood of overpayment. The magnitude indicates its contribution.")
print("  - **Importance:** The absolute value of the coefficient, indicating how much that term influences the prediction, regardless of direction.")
print("  - **Support (for rules):** The proportion of training samples for which the rule conditions were true. This indicates how frequently the rule's pattern occurs.")
print("- **Actionable Use:**")
print("  - **Combined Insights:** RuleFit provides a holistic view. You might find that `Total_Charges` has a direct linear effect, but also specific *combinations* of `DRG_Code` and `Length_of_Stay` (as rules) are highly indicative.")
print("  - **Quantified Impact:** The coefficients give a sense of the *strength* of each pattern. A rule with a high positive coefficient is a strong indicator of overpayment.")
print("  - **Policy Formulation:** Identify the top rules and linear terms. These represent the key patterns associated with overpayment and can inform billing policy updates or audit criteria.")
print("  - **Risk Scoring:** The model's prediction itself can be used as a risk score, and the contributing rules/features explain that score for each case.")

print("\n**General Best Practices:**")
print("1.  **Domain Expertise:** Always review extracted rules with subject matter experts (auditors, clinicians, billing specialists). They can validate whether a rule makes clinical/business sense or indicates a previously unknown loophole.")
print("2.  **Iterative Refinement:** These tools help generate hypotheses. Test them on new data, gather feedback, and refine your rules/models.")
print("3.  **Thresholding:** For SkopeRules, experiment with `precision_min` and `recall_min` to balance the number and quality of rules. For RuleFit, the Lasso regularization automatically handles feature selection based on importance.")
print("4.  **Data Quality:** The quality of the rules is directly dependent on the quality and representativeness of your training data.")

print("\nBy using these techniques, you move beyond just identifying overpayments to truly *understanding why* they occur, enabling proactive prevention and more targeted auditing.")
