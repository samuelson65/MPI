import pandas as pd
import networkx as nx
import plotly.graph_objects as go

def build_query_graph(result_df):
    """
    Build a graph from clustered queries.
    Each query is a node, and edges are drawn if queries belong to the same cluster.

    Parameters
    ----------
    result_df : pd.DataFrame
        Must have columns ["query_name", "cluster_id"]

    Returns
    -------
    G : networkx.Graph
        Graph of queries connected within clusters
    """

    G = nx.Graph()

    # Add nodes with cluster info
    for _, row in result_df.iterrows():
        G.add_node(row["query_name"], cluster=row["cluster_id"])

    # Add edges: connect all queries inside the same cluster
    for cluster_id, group in result_df.groupby("cluster_id"):
        queries = group["query_name"].tolist()
        for i in range(len(queries)):
            for j in range(i + 1, len(queries)):
                G.add_edge(queries[i], queries[j], cluster=cluster_id)

    return G


def plot_query_graph(G):
    """
    Plot query graph using Plotly.
    """

    pos = nx.spring_layout(G, seed=42)  # nice stable layout
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color="#888"),
        hoverinfo="none",
        mode="lines"
    )

    node_x, node_y, node_color, text = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(G.nodes[node]["cluster"])
        text.append(f"{node}<br>Cluster: {G.nodes[node]['cluster']}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[n for n in G.nodes()],
        textposition="bottom center",
        hovertext=text,
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            color=node_color,
            size=18,
            colorbar=dict(
                thickness=15,
                title="Cluster ID",
                xanchor="left",
                titleside="right"
            ),
            line=dict(width=2, color="DarkSlateGrey")
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Query Similarity Graph (Clustered)",
                        title_x=0.5,
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=20, l=5, r=5, t=40)
                    ))

    fig.show()


# --------------------------
# 🔹 Example usage
if __name__ == "__main__":
    # Example clustering result_df
    result_df = pd.DataFrame({
        "query_name": ["A", "B", "C", "D", "E", "F"],
        "cluster_id": [1, 1, 1, 2, 2, 3]
    })

    G = build_query_graph(result_df)
    plot_query_graph(G)
