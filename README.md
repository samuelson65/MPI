import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Example: Define your features (replace with your actual column names)
categorical_features = ['DRG', 'Provider_Type']  # Example categorical columns
numerical_features = ['Claim_Amount', 'Days_Submitted']  # Example numerical columns

# Combine features (replace with your actual DataFrame)
# X_train = pd.read_csv('your_X_train.csv')
# y_train = pd.read_csv('your_y_train.csv').values.ravel()  # Ensure y_train is 1D

# 1. Encode categorical variables
encoder = OrdinalEncoder()
X_cat = encoder.fit_transform(X_train[categorical_features])
X_num = X_train[numerical_features].values
import numpy as np
X_all = np.hstack([X_cat, X_num])

# 2. Train decision tree
dtree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=42)
dtree.fit(X_all, y_train)

# 3. Visualize the decision tree
feature_names = list(encoder.get_feature_names_out(categorical_features)) + numerical_features

plt.figure(figsize=(20, 10))
plot_tree(
    dtree,
    feature_names=feature_names,
    class_names=['Valid', 'Overpayment'],
    filled=True,
    rounded=True,
    proportion=True,
    impurity=False
)
plt.title("Medicare DRG Overpayment Decision Tree", fontsize=20)
plt.show()
