import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Sample diagnosis code data
sample_data = {
    'diagnosis_codes': [
        "E11.9,I10,J45.909",      # Diabetes, Hypertension, Asthma
        "E11.9,I10",              # Diabetes, Hypertension
        "I10,J45.909",            # Hypertension, Asthma
        "E11.9,I10",              # Diabetes, Hypertension
        "J45.909",                # Asthma
        "E11.9",                  # Diabetes
        "I10",                    # Hypertension
        "E11.9,I10,J45.909",      # All three again
        "E11.9,J45.909",          # Diabetes, Asthma
        "I10,J45.909"             # Hypertension, Asthma
    ]
}

# Step 2: Create DataFrame
df = pd.DataFrame(sample_data)

# Step 3: Split diagnosis codes into list per row
transactions = df['diagnosis_codes'].apply(lambda x: x.split(',')).tolist()

# Step 4: One-hot encoding
te = TransactionEncoder()
te_data = te.fit_transform(transactions)
df_encoded = pd.DataFrame(te_data, columns=te.columns_)

# Step 5: Frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=0.3, use_colnames=True)

# Step 6: Association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Show top rules
print("Top Association Rules:\n")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Step 7: Visualize with NetworkX
# Filter rules for clarity
filtered_rules = rules[(rules['confidence'] > 0.6) & (rules['lift'] > 1.0)]

# Create graph
G = nx.DiGraph()

# Add edges
for _, row in filtered_rules.iterrows():
    for antecedent in row['antecedents']:
        for consequent in row['consequents']:
            G.add_edge(antecedent, consequent, weight=row['lift'], confidence=row['confidence'])

# Layout and draw
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.5)

# Draw nodes and edges
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, arrows=True)
edge_labels = { (u,v): f"lift={d['weight']:.2f}" for u,v,d in G.edges(data=True) }
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

plt.title("Diagnosis Code Association Rules (Network Graph)")
plt.show()
