import pandas as pd
import networkx as nx
import community.community_louvain as community_louvain
from py2neo import Graph, Node, Relationship

# ----------------------------
# 1. Input DataFrame
# ----------------------------
# Example data
data = {
    "query_name": ["QueryA", "QueryA", "QueryA", "QueryB", "QueryB",
                   "QueryC", "QueryC", "QueryD", "QueryD", "QueryE"],
    "claim_id":  ["C1", "C2", "C3", "C3", "C4",
                  "C6", "C7", "C6", "C8", "C1"]
}
df = pd.DataFrame(data)

# ----------------------------
# 2. Build Claim Sets per Query
# ----------------------------
query_claims = df.groupby("query_name")["claim_id"].apply(set).to_dict()

# ----------------------------
# 3. Compute Jaccard Similarity
# ----------------------------
def jaccard(set1, set2):
    if not set1 or not set2:
        return 0
    return len(set1 & set2) / len(set1 | set2)

def build_graph(query_claims, threshold=0.1):
    G = nx.Graph()
    queries = list(query_claims.keys())

    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            q1, q2 = queries[i], queries[j]
            sim = jaccard(query_claims[q1], query_claims[q2])
            if sim >= threshold:  # adjustable threshold
                G.add_edge(q1, q2, weight=sim)
    return G

# ----------------------------
# 4. Louvain Clustering
# ----------------------------
def cluster_queries(query_claims, threshold=0.1):
    G = build_graph(query_claims, threshold)
    if len(G.edges) == 0:
        return {q: i for i, q in enumerate(query_claims.keys())}  # each its own cluster
    partition = community_louvain.best_partition(G, weight="weight")
    return partition

partition = cluster_queries(query_claims, threshold=0.2)

# ----------------------------
# 5. Output DataFrame
# ----------------------------
cluster_df = pd.DataFrame(list(partition.items()), columns=["query_name", "cluster_id"])
print(cluster_df)

# ----------------------------
# 6. Push to Neo4j (Optional)
# ----------------------------
# Connect to local Neo4j (make sure Neo4j is running and replace credentials)
try:
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

    # Clear old data
    graph.delete_all()

    # Add nodes
    nodes = {}
    for query in query_claims.keys():
        node = Node("Query", name=query, cluster=partition.get(query, -1))
        graph.create(node)
        nodes[query] = node

    # Add relationships with similarity as weight
    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            q1, q2 = queries[i], queries[j]
            sim = jaccard(query_claims[q1], query_claims[q2])
            if sim >= 0.2:  # same threshold
                rel = Relationship(nodes[q1], "SIMILAR_TO", nodes[q2], weight=sim)
                graph.create(rel)

    print("✅ Graph pushed to Neo4j successfully!")

except Exception as e:
    print("⚠️ Neo4j connection skipped:", e)
