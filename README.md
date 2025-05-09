import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import matplotlib.pyplot as plt

# -------- STEP 1: Load Data --------
# Option 1: Load from CSV
# df = pd.read_csv("your_file.csv")  # Ensure it has a 'diagnosis_codes' column

# Option 2: Sample data if you don’t have a CSV
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
df = pd.DataFrame(sample_data)

# -------- STEP 2: Preprocess --------
transactions = df['diagnosis_codes'].apply(lambda x: x.split(',')).tolist()

te = TransactionEncoder()
te_data = te.fit_transform(transactions)
df_encoded = pd.DataFrame(te_data, columns=te.columns_)

# -------- STEP 3: Mine Rules --------
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

# -------- STEP 4: Export to Excel --------
if not rules.empty:
    rules_to_export = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
    
    # Convert frozensets to strings for Excel compatibility
    rules_to_export['antecedents'] = rules_to_export['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules_to_export['consequents'] = rules_to_export['consequents'].apply(lambda x: ', '.join(list(x)))
    
    # Export to Excel
    rules_to_export.to_excel("association_rules_output.xlsx", index=False)
    print("Rules exported to 'association_rules_output.xlsx'")
else:
    print("No association rules found to export.")

# -------- STEP 5: Optional Visualization --------
if not rules.empty:
    filtered_rules = rules[(rules['confidence'] > 0.1) & (rules['lift'] > 1.0)]

    G = nx.DiGraph()
    for _, row in filtered_rules.iterrows():
        for antecedent in row['antecedents']:
            for consequent in row['consequents']:
                G.add_edge(antecedent, consequent, weight=row['lift'], confidence=row['confidence'])

    if len(G) > 0:
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.5)
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightgreen', font_size=10, arrows=True)
        edge_labels = { (u,v): f"lift={d['weight']:.2f}" for u,v,d in G.edges(data=True) }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkred')
        plt.title("Diagnosis Code Association Rules (Network Graph)")
        plt.show()
    else:
        print("No strong associations to visualize.")
