import pandas as pd
import itertools
import networkx as nx

def compute_overlap(setA, setB):
    """Overlap coefficient = |A ∩ B| / min(|A|,|B|)"""
    A, B = set(setA), set(setB)
    inter = len(A & B)
    denom = min(len(A), len(B))
    return inter / denom if denom > 0 else 0.0

def cluster_queries(df, overlap_threshold=0.8):
    """
    Cluster queries based on overlap coefficient threshold.
    Input: df with columns [query_name, claim_id]
    Output: DataFrame with query_name and cluster_id
    """
    # group claims by query
    query_claims = df.groupby("query_name")["claim_id"].apply(set).to_dict()

    # build graph
    G = nx.Graph()
    G.add_nodes_from(query_claims.keys())

    for (q1, claims1), (q2, claims2) in itertools.combinations(query_claims.items(), 2):
        overlap = compute_overlap(claims1, claims2)
        if overlap >= overlap_threshold:
            G.add_edge(q1, q2, weight=overlap)

    # get connected components (clusters)
    clusters = list(nx.connected_components(G))
    cluster_map = {}
    for idx, comp in enumerate(clusters, start=1):
        for q in comp:
            cluster_map[q] = idx

    # build output df
    out_df = pd.DataFrame({
        "query_name": list(cluster_map.keys()),
        "cluster_id": list(cluster_map.values())
    })

    return out_df.sort_values("cluster_id").reset_index(drop=True)


# -------------------------------
# 🔹 Example usage
if __name__ == "__main__":
    data = [
        ("A", "C1"), ("A", "C2"), ("A", "C3"),
        ("B", "C1"), ("B", "C2"), ("B", "C3"), ("B", "C4"),
        ("C", "C2"), ("C", "C3"), ("C", "C5"),
        ("D", "C10"), ("D", "C11"),
        ("E", "C10"), ("E", "C11"), ("E", "C12")
    ]
    df = pd.DataFrame(data, columns=["query_name", "claim_id"])

    clustered_df = cluster_queries(df, overlap_threshold=0.8)

    print("\nClustered Queries:")
    print(clustered_df)
