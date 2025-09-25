import pandas as pd
import networkx as nx

def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def overlap_coefficient(set1, set2):
    """Compute Overlap Coefficient = |A ∩ B| / min(|A|, |B|)."""
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / min(len(set1), len(set2))

def cluster_queries(result_df, threshold=0.8, metric="overlap"):
    """
    Cluster queries based on claim overlap.

    Parameters
    ----------
    result_df : pd.DataFrame
        Must contain ["query_name", "claim_id"]
    threshold : float
        Similarity threshold (default 0.8)
    metric : str
        "jaccard" or "overlap"

    Returns
    -------
    clustered_df : pd.DataFrame
        ["query_name", "cluster_id"]
    """

    # build dictionary: query_name -> set of claims
    query_claims = result_df.groupby("query_name")["claim_id"].apply(set).to_dict()

    # build graph of queries
    G = nx.Graph()
    for q in query_claims.keys():
        G.add_node(q)

    queries = list(query_claims.keys())
    for i in range(len(queries)):
        for j in range(i+1, len(queries)):
            q1, q2 = queries[i], queries[j]
            set1, set2 = query_claims[q1], query_claims[q2]

            if metric == "jaccard":
                score = jaccard_similarity(set1, set2)
            else:
                score = overlap_coefficient(set1, set2)

            if score >= threshold:
                G.add_edge(q1, q2, weight=score)

    # connected components = clusters
    clusters = list(nx.connected_components(G))

    cluster_map = {}
    for cluster_id, nodes in enumerate(clusters, start=1):
        for q in nodes:
            cluster_map[q] = cluster_id

    clustered_df = pd.DataFrame([
        {"query_name": q, "cluster_id": cluster_map[q]} for q in queries
    ])

    return clustered_df


# ------------------------------
# 🔹 Example usage
if __name__ == "__main__":
    data = {
        "query_name": ["A", "A", "B", "B", "C", "D", "D", "E"],
        "claim_id":   [101, 102, 101, 103, 104, 105, 106, 105]
    }
    df = pd.DataFrame(data)

    clustered_df = cluster_queries(df, threshold=0.8, metric="overlap")
    print(clustered_df)
