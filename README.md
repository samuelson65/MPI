import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Load data
df_raw = pd.read_csv("diagnosis_data.csv")  # replace with your actual file name

# Step 2: Split diagnosis codes into lists
transactions = df_raw['diagnosis_codes'].apply(lambda x: x.split(',')).tolist()

# Step 3: Convert to one-hot encoded format
te = TransactionEncoder()
te_data = te.fit_transform(transactions)
df_encoded = pd.DataFrame(te_data, columns=te.columns_)

# Step 4: Generate frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=0.3, use_colnames=True)

# Step 5: Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Step 6: Display important rules
rules_sorted = rules.sort_values(by="confidence", ascending=False)
print("Top Association Rules:\n")
print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

import networkx as nx
import matplotlib.pyplot as plt

# Filter rules with good confidence and lift for clearer visualization
filtered_rules = rules[(rules['confidence'] > 0.6) & (rules['lift'] > 1.0)]

# Create a graph
G = nx.DiGraph()

# Add edges from antecedents to consequents
for _, row in filtered_rules.iterrows():
    for antecedent in row['antecedents']:
        for consequent in row['consequents']:
            G.add_edge(antecedent, consequent, weight=row['lift'], confidence=row['confidence'])

# Position nodes using spring layout
pos = nx.spring_layout(G, k=0.5, iterations=20)

# Draw nodes and edges
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)

# Add edge labels (lift)
edge_labels = nx.get_edge_attributes(G, 'lift')
edge_labels = {k: f"lift={v:.2f}" for k, v in edge_labels.items()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

plt.title("Diagnosis Code Association Rules (Network Graph)")
plt.show()
