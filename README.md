"""
Scalable Query Clustering with Louvain
- Efficient Jaccard via inverted index (claim -> queries that hit it)
- Thresholding + top-K sparsification for large graphs
- Optional: Push to Neo4j GDS for clustering at scale
- Optional: Plotly visualization on a sampled subgraph

INPUT assumptions:
  - A CSV (or DataFrame) with at least: claimid, query_name
  - Optional columns: weight (e.g., claim paid amount) to build cost-weighted Jaccard

Install:
  pip install pandas numpy networkx python-louvain plotly neo4j tqdm

If you have very large CSVs, make sure to run with enough RAM or increase chunk_size below.
"""

import os
import math
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
from tqdm import tqdm
import networkx as nx
import plotly.graph_objects as go
import community as community_louvain  # pip install python-louvain

# -----------------------------
# CONFIG
# -----------------------------
INPUT_CSV = None      # e.g., "claims_queries.csv"  # if None, we'll use a sample DF
CHUNK_SIZE = 2_000_000  # rows per chunk (tune to your memory)
JACCARD_THRESHOLD = 0.15  # edge created only if >= this similarity
TOPK_PER_QUERY = 20       # cap edges per query to keep the graph sparse
USE_WEIGHTED_JACCARD = False  # if True, weight by claim dollars (column name 'claim_weight')
CLAIM_WEIGHT_COL = "claim_weight"

# For Plotly visualization (sample a subgraph to avoid heavy plotting)
PLOT_SAMPLE_MAX_NODES = 300  # set lower for very large graphs
SEED = 42


# -----------------------------
# 1) LOAD DATA (chunked or DataFrame)
#    Expect columns: claimid, query_name [, claim_weight]
# -----------------------------
def load_data(input_csv: str | None) -> pd.DataFrame:
    if input_csv is None:
        # Sample data for demonstration
        data = {
            "claimid": [
                "C1","C2","C3","C3","C4","C5","C5","C6","C7","C8","C9","C10","C10","C11","C12","C13","C13","C13","C14"
            ],
            "query_name": [
                "A","A","A","B","B","B","C","C","D","D","E","F","A","G","H","I","B","E","J"
            ],
            # Optional: monetary weight per claim row (could be paid amount)
            CLAIM_WEIGHT_COL: [1000,1500,1200,1200,800,2000,2500,1800,2200,3000,1700,900,1100,500,450,700,650,600,400]
        }
        return pd.DataFrame(data)
    else:
        # If data is huge, consider returning an iterator (but we need multiple passes below)
        # For simplicity, we read fully; for massive data, see the chunked inverted-index builder further below.
        return pd.read_csv(input_csv)


# -----------------------------
# 2) BUILD INVERTED INDEX (claim -> set/list of queries)
#    Also track per-query supports and (optional) weighted supports.
#    This scales well for large Q since we iterate only co-occurring pairs per claim.
# -----------------------------
def build_inverted_index(df: pd.DataFrame,
                         use_weighted: bool = False,
                         weight_col: str = CLAIM_WEIGHT_COL):
    """
    Returns:
        claim_to_queries: dict(claimid -> list of queries)
        query_support_count: dict(query -> number of unique claims)
        query_weight_sum: dict(query -> sum of weights)  (if use_weighted)
    """
    # Ensure the key columns exist
    assert {"claimid", "query_name"}.issubset(df.columns), "DataFrame must have claimid, query_name"
    if use_weighted:
        assert weight_col in df.columns, f"Weighted mode requires '{weight_col}' column"

    # Deduplicate to avoid counting the same (claim, query) twice
    # (if your data guarantees uniqueness, you can skip this to save time)
    df = df[["claimid", "query_name"] + ([weight_col] if use_weighted else [])].drop_duplicates()

    # Claim -> queries
    claim_to_queries = defaultdict(list)
    if use_weighted:
        # store the weight per (claim, query) if provided (often weight is per claim, but we’ll accept per-row)
        row_weight = {}
        for row in df.itertuples(index=False):
            c, q, w = row
            claim_to_queries[c].append(q)
            row_weight[(c, q)] = float(w)
    else:
        for c, q in df[["claimid", "query_name"]].itertuples(index=False):
            claim_to_queries[c].append(q)
        row_weight = None

    # Query supports (#claims) and weighted supports
    # Use sets for unique claims per query
    query_claims = defaultdict(set)
    if use_weighted:
        query_weight_sum = defaultdict(float)
        for (c, q), w in (row_weight.items()):
            query_claims[q].add(c)
            query_weight_sum[q] += float(w)
    else:
        query_weight_sum = None
        for c, q in df[["claimid", "query_name"]].itertuples(index=False):
            query_claims[q].add(c)

    query_support_count = {q: len(s) for q, s in query_claims.items()}
    return claim_to_queries, query_support_count, query_weight_sum, row_weight


# -----------------------------
# 3) EFFICIENT JACCARD OVERLAP COUNTS
#    For each claim, increment intersection counts for all query pairs that co-occur on that claim.
# -----------------------------
def compute_intersections(claim_to_queries: dict[str, list[str]]) -> Counter:
    """
    Builds a Counter of pair intersections using the inverted index:
      For each claim: for each pair of queries that selected it, increment their intersection by 1.
    Returns Counter with key as tuple(sorted(q1,q2)), value = intersection count.
    """
    intersections = Counter()
    for queries in claim_to_queries.values():
        if len(queries) < 2:
            continue
        # de-duplicate per-claim query list to avoid double counting same (q1,q2) for repeated rows
        uq = sorted(set(queries))
        for i in range(len(uq)):
            for j in range(i + 1, len(uq)):
                intersections[(uq[i], uq[j])] += 1
    return intersections


# -----------------------------
# 4) (Optional) WEIGHTED INTERSECTIONS
#    If using monetary weights, sum min-weights on co-occurring claim edges.
#    A simple approach: per claim, for each (q1,q2) pair, add min(weight(c,q1), weight(c,q2)) to intersection_weight.
# -----------------------------
def compute_weighted_intersections(claim_to_queries: dict[str, list[str]],
                                   row_weight: dict[tuple, float]) -> Counter:
    weighted_intersections = Counter()
    for c, queries in claim_to_queries.items():
        if len(queries) < 2:
            continue
        uq = sorted(set(queries))
        for i in range(len(uq)):
            for j in range(i + 1, len(uq)):
                q1, q2 = uq[i], uq[j]
                w1 = row_weight.get((c, q1), 0.0)
                w2 = row_weight.get((c, q2), 0.0)
                weighted_intersections[(q1, q2)] += min(w1, w2)
    return weighted_intersections


# -----------------------------
# 5) BUILD EDGE LIST WITH JACCARD (normal or weighted)
# -----------------------------
def build_edges(intersections: Counter,
                query_support_count: dict[str, int],
                weighted_intersections: Counter | None = None,
                query_weight_sum: dict[str, float] | None = None,
                jaccard_threshold: float = JACCARD_THRESHOLD,
                topk_per_query: int = TOPK_PER_QUERY):
    """
    Returns a list of (q1, q2, weight) edges.
    If weighted_intersections and query_weight_sum are provided, compute cost-weighted Jaccard:
      Jaccard_w = sum_min_weights / (sum_w(q1) + sum_w(q2) - sum_min_weights)
    Else compute standard Jaccard on claim counts.
    Also applies per-node TOP-K pruning to keep the graph sparse.
    """
    # Step 1: compute raw similarity for all pairs
    neighbors = defaultdict(list)  # q -> list of (other, sim)
    for (q1, q2), inter in intersections.items():
        if weighted_intersections is not None and query_weight_sum is not None:
            inter_w = weighted_intersections.get((q1, q2), 0.0)
            denom = query_weight_sum[q1] + query_weight_sum[q2] - inter_w
            if denom <= 0:
                continue
            sim = inter_w / denom
        else:
            # vanilla Jaccard
            denom = query_support_count[q1] + query_support_count[q2] - inter
            if denom <= 0:
                continue
            sim = inter / denom

        if sim >= jaccard_threshold:
            neighbors[q1].append((q2, sim))
            neighbors[q2].append((q1, sim))

    # Step 2: per-node TOP-K pruning (keep strongest K edges)
    pruned_edges = set()
    for q, lst in neighbors.items():
        lst.sort(key=lambda x: x[1], reverse=True)
        for other, sim in lst[:topk_per_query]:
            key = tuple(sorted((q, other)))
            pruned_edges.add((key[0], key[1], sim))

    # Convert to list
    edges = list(pruned_edges)
    return edges


# -----------------------------
# 6) LOUVAIN CLUSTERING (LOCAL)
# -----------------------------
def louvain_cluster(edges: list[tuple[str, str, float]]):
    """
    Build a NetworkX weighted graph and run Louvain clustering.
    Returns:
        partition: dict(query -> community_id)
        G: the NetworkX graph
    """
    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=float(w))
    if G.number_of_nodes() == 0:
        return {}, G

    partition = community_louvain.best_partition(G, weight="weight", random_state=SEED)
    return partition, G


# -----------------------------
# 7) PLOTLY INTERACTIVE GRAPH (sampled)
# -----------------------------
def plotly_graph(G: nx.Graph, partition: dict[str, int], max_nodes=PLOT_SAMPLE_MAX_NODES, title="Query Graph (Louvain)"):
    if G.number_of_nodes() == 0:
        print("Graph is empty; nothing to plot.")
        return

    # Sample if too large
    if G.number_of_nodes() > max_nodes:
        # take the largest connected component, then head(max_nodes)
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        nodes = list(next(iter(comps)))
        # if still large, downsample by degree
        if len(nodes) > max_nodes:
            degrees = G.degree(nodes)
            nodes = [n for n, _ in sorted(degrees, key=lambda kv: kv[1], reverse=True)[:max_nodes]]
        H = G.subgraph(nodes).copy()
        part = {n: partition[n] for n in H.nodes()}
    else:
        H = G.copy()
        part = partition

    pos = nx.spring_layout(H, seed=SEED, weight="weight", k=1/math.sqrt(max(1, H.number_of_nodes())))

    # Edges
    edge_x, edge_y, edge_text = [], [], []
    for u, v, d in H.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
        edge_text.append(f"{u}—{v}<br>sim={d.get('weight', 0):.3f}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=1), hoverinfo='text',
        text=edge_text, opacity=0.5
    )

    # Nodes
    node_x, node_y, node_color, node_text = [], [], [], []
    for n in H.nodes():
        x, y = pos[n]
        node_x.append(x); node_y.append(y)
        node_color.append(part.get(n, -1))
        # degree-weighted tooltip helps spot central queries
        node_text.append(f"{n}<br>cluster={part.get(n, -1)}<br>deg={H.degree(n)}")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=[n for n in H.nodes()],
        textposition='bottom center',
        hovertext=node_text, hoverinfo='text',
        marker=dict(size=12, color=node_color, showscale=True, colorbar=dict(title="Cluster")),
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=title, titlefont_size=18,
                        showlegend=False, hovermode='closest',
                        margin=dict(b=20, l=20, r=20, t=60),
                        xaxis=dict(showgrid=False, zeroline=False, visible=False),
                        yaxis=dict(showgrid=False, zeroline=False, visible=False)
                    ))
    fig.show()


# -----------------------------
# 8) NEO4J GDS PATH (OPTIONAL)
# -----------------------------
def neo4j_push_and_louvain(nodes: list[str], edges: list[tuple[str, str, float]],
                           neo4j_uri: str, neo4j_user: str, neo4j_pwd: str,
                           graph_name: str = "queryGraph", write_property: str = "louvain"):
    """
    Push nodes & edges to Neo4j and run GDS Louvain.
    Requires:
      - Neo4j + GDS plugin installed
      - pip install neo4j
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pwd))
    with driver.session() as session:
        # Clean previous data
        session.run("MATCH (n:Query) DETACH DELETE n")

        # Create nodes
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (q:Query) REQUIRE q.id IS UNIQUE")
        for q in nodes:
            session.run("MERGE (q:Query {id:$id})", id=q)

        # Create edges with weight
        for u, v, w in edges:
            session.run("""
                MATCH (a:Query {id:$u}), (b:Query {id:$v})
                MERGE (a)-[r:SIMILAR_TO]->(b)
                SET r.weight = $w
            """, u=u, v=v, w=float(w))

        # Project graph into GDS
        session.run(f"CALL gds.graph.drop('{graph_name}', false) YIELD graphName RETURN graphName")
        session.run(f"""
            CALL gds.graph.project(
                '{graph_name}',
                'Query',
                {{ SIMILAR_TO: {{ orientation: 'UNDIRECTED', properties: 'weight' }} }}
            )
        """)

        # Run Louvain and write back the community id
        session.run(f"""
            CALL gds.louvain.write('{graph_name}', {{
              relationshipWeightProperty: 'weight',
              writeProperty: '{write_property}'
            }})
            YIELD communityCount, modularity, modularities
        """)

        # Retrieve results
        result = session.run(f"MATCH (q:Query) RETURN q.id AS id, q.{write_property} AS community")
        partition = {rec["id"]: rec["community"] for rec in result}

    driver.close()
    return partition


# -----------------------------
# 9) MAIN: tie it together
# -----------------------------
def run_pipeline(input_csv=INPUT_CSV,
                 jaccard_threshold=JACCARD_THRESHOLD,
                 topk_per_query=TOPK_PER_QUERY,
                 use_weighted=USE_WEIGHTED_JACCARD):
    df = load_data(input_csv)

    # Build inverted index + supports
    claim_to_queries, query_support_count, query_weight_sum, row_weight = build_inverted_index(
        df, use_weighted=use_weighted, weight_col=CLAIM_WEIGHT_COL
    )

    # Intersections
    intersections = compute_intersections(claim_to_queries)

    # Weighted intersections (optional)
    if use_weighted:
        weighted_intersections = compute_weighted_intersections(claim_to_queries, row_weight)
    else:
        weighted_intersections = None

    # Build edges w/ Jaccard or weighted Jaccard; prune with top-k
    edges = build_edges(
        intersections,
        query_support_count,
        weighted_intersections,
        query_weight_sum,
        jaccard_threshold=jaccard_threshold,
        topk_per_query=topk_per_query
    )

    # Local Louvain
    partition, G = louvain_cluster(edges)

    # Results table
    part_df = pd.DataFrame([(q, cid) for q, cid in partition.items()], columns=["query_name", "cluster"])
    print("Clusters (local Louvain):")
    print(part_df.sort_values(["cluster", "query_name"]).to_string(index=False))

    # Optional: Plotly visualization (sampled)
    plotly_graph(G, partition, max_nodes=PLOT_SAMPLE_MAX_NODES, title="Query Graph (Louvain)")

    return part_df, edges, G


# -----------------------------
# 10) RUN
# -----------------------------
if __name__ == "__main__":
    clusters_df, edges, G = run_pipeline(
        input_csv=INPUT_CSV,
        jaccard_threshold=0.15,   # tune
        topk_per_query=20,        # tune
        use_weighted=False        # set True if you have claim dollars & want cost-weighted Jaccard
    )

    # --- OPTIONAL: Push to Neo4j (uncomment and set creds) ---
    # NEO4J_URI = "bolt://localhost:7687"
    # NEO4J_USER = "neo4j"
    # NEO4J_PWD  = "password"
    # neo4j_partition = neo4j_push_and_louvain(
    #     nodes=list(G.nodes()),
    #     edges=[(u, v, d["weight"]) for u, v, d in G.edges(data=True)],
    #     neo4j_uri=NEO4J_URI, neo4j_user=NEO4J_USER, neo4j_pwd=NEO4J_PWD,
    #     graph_name="queryGraph", write_property="louvain"
    # )
    # print("Neo4j Louvain communities (sample):", list(neo4j_partition.items())[:10])
