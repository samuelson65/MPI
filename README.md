import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import graphviz
import os

# --- 1. Create Dummy Data (Replace with your actual X_train and y_train) ---
# In a real-world scenario, you would load your pre-split X_train and y_train here.
# For demonstration purposes, let's create a synthetic dataset that resembles DRG Medicare data.
# This dataset includes categorical features ('DRG_Code', 'Provider_Type', 'Diagnosis_Group')
# and numerical features ('Length_of_Stay', 'Total_Charges').
# 'Overpayment' is our target variable (1 for overpayment, 0 for no overpayment).
data = {
    'DRG_Code': ['291', '292', '291', '871', '291', '871', '292', '872', '291', '871'],
    'Provider_Type': ['Hospital', 'Hospital', 'Skilled Nursing', 'Hospital', 'Hospital',
                      'Skilled Nursing', 'Hospital', 'Hospital', 'Hospital', 'Skilled Nursing'],
    'Length_of_Stay': [5, 10, 3, 15, 6, 4, 12, 7, 8, 5],
    'Total_Charges': [10000, 25000, 5000, 30000, 12000, 6000, 28000, 15000, 13000, 7000],
    'Diagnosis_Group': ['Heart Failure', 'Pneumonia', 'Heart Failure', 'Stroke', 'Heart Failure',
                        'Stroke', 'Pneumonia', 'Diabetes', 'Heart Failure', 'Stroke'],
    'Overpayment': [0, 1, 0, 1, 0, 0, 1, 0, 0, 0] # 1 for overpayment, 0 for no overpayment
}
df = pd.DataFrame(data)

# Separate features (X) and target (y)
X = df.drop('Overpayment', axis=1)
y = df['Overpayment']

# For this example, we'll split the dummy data into training and testing sets.
# In your actual use case, you'd likely already have your X_train, y_train.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Identify Categorical and Numerical Features ---
# This step is crucial for applying the correct preprocessing.
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns

# --- 3. Preprocessing Pipeline for Categorical Variables (One-Hot Encoding) ---
# We use a ColumnTransformer to apply OneHotEncoder to only the categorical features.
# 'handle_unknown='ignore'' is important for unseen categories in new data (e.g., test set).
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough' # This keeps the numerical features as they are.
)

# --- 4. Build and Train the Decision Tree Classifier ---
# A Scikit-learn Pipeline is used to chain the preprocessing and the model training.
# This ensures that preprocessing steps are consistently applied.
# We use `DecisionTreeClassifier` with a `random_state` for reproducibility.
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', DecisionTreeClassifier(random_state=42, max_depth=5))])
# `max_depth` is set to 5 to prevent overfitting and make the tree more interpretable.
# You might adjust this based on your data.

# Train the model using the training data.
print("Training the Decision Tree model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# To get the feature names after one-hot encoding, we need to fit the preprocessor
# and then access the names from the fitted OneHotEncoder.
preprocessor.fit(X_train) # Fit the preprocessor separately to get feature names
encoded_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = list(encoded_feature_names) + list(numerical_features)

# --- 5. Visualize the Decision Tree ---
# Access the trained DecisionTreeClassifier instance from within the pipeline.
decision_tree_model = model_pipeline.named_steps['classifier']

# Export the decision tree into DOT format, which can be rendered by Graphviz.
dot_data = export_graphviz(decision_tree_model,
                           out_file=None, # Output as string
                           feature_names=all_feature_names, # Names of features for clarity
                           class_names=['No Overpayment', 'Overpayment'], # Names of target classes
                           filled=True, # Color nodes to indicate class
                           rounded=True, # Round node corners
                           special_characters=True) # Handle special characters in node labels

# Create a Graphviz Source object from the DOT data.
graph = graphviz.Source(dot_data)

# Define the output file name and format.
output_file_name = "drg_overpayment_decision_tree"

# Render the graph to a PDF file. `view=False` prevents the file from opening automatically.
try:
    graph.render(output_file_name, format='pdf', view=False)
    print(f"\nDecision tree visualization saved to {output_file_name}.pdf")
except Exception as e:
    print(f"\nError rendering the decision tree visualization: {e}")
    print("Please ensure Graphviz is installed on your system and its executable is in your PATH.")
    print("You can download it from: https://graphviz.org/download/")
    print("And install the Python library: pip install graphviz")

# --- 6. Guidance for Subject Matter Experts (SMEs) ---
print("\n" + "="*50)
print("--- Guidance for Subject Matter Experts (SMEs) ---")
print("="*50)
print("\nThe generated PDF file ('drg_overpayment_decision_tree.pdf') provides a visual representation of how the model identifies potential overpayments.")
print("Each **node** in the tree represents a decision rule based on a specific feature and a threshold. Following the paths from the top (root) down to the bottom (leaf nodes) reveals the logical steps the model takes.")

print("\n### How to Interpret and Develop Concepts/Queries:")
print("1.  **Follow the Decision Paths**: Start at the root node (the very top). Each split (branch) indicates a condition (e.g., `Length_of_Stay <= 7.5`). If the condition is true, follow the left branch; otherwise, follow the right branch. Continue until you reach a **leaf node** (a node with no further branches).")
print("    * **Leaf Nodes**: These nodes represent the final prediction. Look at the `class` label to see if it predicts 'Overpayment' or 'No Overpayment'.")
print("    * The `value` array within each node shows the count of samples for each class at that point (e.g., `[no_overpayment_count, overpayment_count]`).")

print("\n2.  **Identify Key Features**: Observe which features appear higher up in the tree. These are the most influential factors in determining overpayment according to the model.")
print("    * For instance, if `DRG_Code_292` or `Total_Charges` appear early, they are strong indicators.")

print("\n3.  **Extract Conditional Rules for Overpayment**: For each path that leads to an 'Overpayment' leaf node, carefully list all the conditions encountered along that path. These conditions form a specific rule for identifying overpayments.")
print("    * **Example Rule**: `IF DRG_Code_292 = 1 AND Length_of_Stay > 9.5 AND Total_Charges > 20000 THEN Overpayment`.")
print("    * **Note on One-Hot Encoded Features**: Features like `DRG_Code_292` are results of one-hot encoding. `DRG_Code_292 <= 0.5` typically means `DRG_Code` is NOT '292', and `DRG_Code_292 > 0.5` typically means `DRG_Code` IS '292'. Pay attention to these binary conditions.")

print("\n4.  **Formulate SQL/Database Queries**: Translate the extracted conditional rules into precise SQL or database queries. These queries can then be run on your actual DRG Medicare dataset to find instances that match the 'overpayment' patterns identified by the model.")
print("    * **Example SQL Query based on the rule above**:")
print("        ```sql")
print("        SELECT *")
print("        FROM Your_DRG_Medicare_Table")
print("        WHERE DRG_Code = '292'")
print("          AND Length_of_Stay > 9.5")
print("          AND Total_Charges > 20000;")
print("        ```")

print("\n5.  **Develop Subject Matter Concepts**: Using these patterns, SMEs can formulate hypotheses about why these specific combinations of features lead to overpayments. This could reveal potential areas of: ")
print("    * **Fraud or Abuse**: Intentional miscoding or inflated charges.")
print("    * **Billing Errors**: Unintentional mistakes in coding or charging.")
print("    * **Operational Inefficiencies**: Longer stays or higher costs without clear medical justification for certain DRGs.")
print("    * **Policy Gaps**: Areas where current policies might be exploited or need refinement.")

print("\n6.  **Refine and Validate**: The insights gained can be used to:")
print("    * **Refine existing audit rules** or create new, targeted audit flags.")
print("    * **Inform policy changes** to close identified loopholes.")
print("    * **Investigate specific providers or cases** that frequently trigger these overpayment rules.")
print("    * **Measure the financial impact** of identified overpayments.")

print("\nBy combining the model's predictive power with your domain expertise, you can effectively identify, understand, and address overpayment issues.")
print("\n" + "="*50)

