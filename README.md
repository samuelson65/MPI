import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

# ---------------------------
# 1. SAMPLE DATA
# ---------------------------
data = {
    'claimid': ['claim_001', 'claim_002', 'claim_002', 'claim_003', 'claim_004', 'claim_005', 'claim_005'],
    'query_name': ['A', 'A', 'B', 'B', 'C', 'A', 'B']
}
df = pd.DataFrame(data)

# ---------------------------
# 2. BUILD QUERY -> CLAIMS MAPPING
# ---------------------------
query_claims = defaultdict(set)
for _, row in df.iterrows():
    query_claims[row['query_name']].add(row['claimid'])

queries = list(query_claims.keys())
claims = sorted(set(df['claimid']))

# ---------------------------
# 3. CREATE BINARY MATRIX FOR COSINE SIMILARITY
# ---------------------------
binary_matrix = pd.DataFrame(0, index=queries, columns=claims)
for query, claimset in query_claims.items():
    binary_matrix.loc[query, list(claimset)] = 1

# ---------------------------
# 4. CALCULATE JACCARD SIMILARITY
# ---------------------------
def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0

jaccard_results = []
for q1, q2 in combinations(queries, 2):
    score = jaccard_similarity(query_claims[q1], query_claims[q2])
    jaccard_results.append((q1, q2, score))

jaccard_df = pd.DataFrame(jaccard_results, columns=["Query_1", "Query_2", "Jaccard_Similarity"])

# ---------------------------
# 5. CALCULATE COSINE SIMILARITY
# ---------------------------
cosine_matrix = cosine_similarity(binary_matrix)
cosine_df = pd.DataFrame(cosine_matrix, index=queries, columns=queries)

cosine_results = []
for q1, q2 in combinations(queries, 2):
    score = cosine_df.loc[q1, q2]
    cosine_results.append((q1, q2, score))

cosine_pairs_df = pd.DataFrame(cosine_results, columns=["Query_1", "Query_2", "Cosine_Similarity"])

# ---------------------------
# 6. PRINT RESULTS
# ---------------------------
print("\n=== Jaccard Similarity ===")
print(jaccard_df)

print("\n=== Cosine Similarity ===")
print(cosine_pairs_df)

# ---------------------------
# 7. BUILD NETWORK GRAPH
# ---------------------------
def plot_network(similarity_df, similarity_type="Jaccard", threshold=0.1):
    G = nx.Graph()
    for _, row in similarity_df.iterrows():
        if row[similarity_type + "_Similarity"] >= threshold:
            G.add_edge(row['Query_1'], row['Query_2'], weight=row[similarity_type + "_Similarity"])

    pos = nx.spring_layout(G, seed=42)
    weights = nx.get_edge_attributes(G, 'weight')

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color="lightblue",
            font_size=12, font_weight="bold", width=2, edge_color="gray")
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in weights.items()},
                                 font_color='red', font_size=10)
    plt.title(f"{similarity_type} Similarity Network (Threshold ≥ {threshold})", fontsize=14)
    plt.show()

# ---------------------------
# 8. PLOT NETWORKS
# ---------------------------
plot_network(jaccard_df.rename(columns={"Jaccard_Similarity": "Jaccard_Similarity"}), "Jaccard", threshold=0.1)
plot_network(cosine_pairs_df.rename(columns={"Cosine_Similarity": "Cosine_Similarity"}), "Cosine", threshold=0.1)
