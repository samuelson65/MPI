import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain  # Louvain
from cdlib import algorithms  # Leiden

# ----------------------------
# 1. Sample Queries and Claims
# ----------------------------
query_claims = {
    "QueryA": {"C1", "C2", "C3", "C4"},
    "QueryB": {"C3", "C4", "C5"},
    "QueryC": {"C6", "C7"},
    "QueryD": {"C6", "C7", "C8"},
    "QueryE": {"C1", "C2", "C3", "C9"}
}

# ----------------------------
# 2. Compute Jaccard Similarity
# ----------------------------
def jaccard(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

# Build Graph
G = nx.Graph()
queries = list(query_claims.keys())

for i in range(len(queries)):
    for j in range(i + 1, len(queries)):
        q1, q2 = queries[i], queries[j]
        sim = jaccard(query_claims[q1], query_claims[q2])
        if sim > 0:  # add edge only if they overlap
            G.add_edge(q1, q2, weight=sim)

# ----------------------------
# 3. Louvain Clustering
# ----------------------------
louvain_partition = community_louvain.best_partition(G, weight="weight")

print("🔹 Louvain Clustering Result:")
for q, cluster in louvain_partition.items():
    print(f"{q} → Cluster {cluster}")

# ----------------------------
# 4. Leiden Clustering
# ----------------------------
leiden_partition = algorithms.leiden(G, weights="weight", resolution=1.0)

print("\n🔹 Leiden Clustering Result:")
for idx, comm in enumerate(leiden_partition.communities):
    print(f"Cluster {idx}: {comm}")

# ----------------------------
# 5. Visualization
# ----------------------------
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(8,6))

# Louvain colors
colors = [louvain_partition[node] for node in G.nodes()]
nx.draw(G, pos, with_labels=True, node_size=1200,
        node_color=colors, cmap=plt.cm.Set3, edge_color="gray")

labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels={k: round(v,2) for k,v in labels.items()})

plt.title("Query Graph with Louvain Clusters")
plt.show()
