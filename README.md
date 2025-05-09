import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Sample diagnosis code data
sample_data = {
    'diagnosis_codes': [
        "E11.9,I10,J45.909",
        "E11.9,I10",
        "I10,J45.909",
        "E11.9,I10",
        "J45.909",
        "E11.9",
        "I10",
        "E11.9,I10,J45.909",
        "E11.9,J45.909",
        "I10,J45.909"
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

# Step 5: Frequent itemsets with lower support for small data
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)

# Step 6: Association rules with relaxed thresholds
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

# Show top rules (don't filter too hard)
print("Top Association Rules:\n")
if not rules.empty:
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
else:
    print("No rules found. Try reducing min_support or min_thresholds further.")

# Step 7: Visualize with NetworkX (only if rules exist)
if not rules.empty:
    # Loosen filters for graph
    filtered_rules = rules[(rules['confidence'] > 0.1) & (rules['lift'] > 1.0)]

    G = nx.DiGraph()
    for _, row in filtered_rules.iterrows():
        for antecedent in row['antecedents']:
            for consequent in row['consequents']:
                G.add_edge(antecedent, consequent, weight=row['lift'], confidence=row['confidence'])

    if len(G) > 0:
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.5)

        nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, arrows=True)
        edge_labels = { (u,v): f"lift={d['weight']:.2f}" for u,v,d in G.edges(data=True) }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        plt.title("Diagnosis Code Association Rules (Network Graph)")
        plt.show()
    else:
        print("No strong associations found to visualize.")
