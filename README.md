import networkx as nx
import community as community_louvain  # Louvain
import plotly.graph_objects as go

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
        if sim > 0:  # only if overlap
            G.add_edge(q1, q2, weight=sim)

# ----------------------------
# 3. Louvain Clustering
# ----------------------------
partition = community_louvain.best_partition(G, weight="weight")

# ----------------------------
# 4. Prepare Plotly Visualization
# ----------------------------
pos = nx.spring_layout(G, seed=42, weight="weight")

# Extract edges for plotting
edge_x = []
edge_y = []
edge_weights = []
for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_weights.append(edge[2]['weight'])

# Edge trace
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1, color='lightgray'),
    hoverinfo='none',
    mode='lines'
)

# Node trace
node_x = []
node_y = []
node_text = []
node_color = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(f"{node}<br>Cluster: {partition[node]}<br>Claims: {list(query_claims[node])}")
    node_color.append(partition[node])  # Louvain cluster ID

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=[n for n in G.nodes()],
    textposition="bottom center",
    hovertext=node_text,
    hoverinfo="text",
    marker=dict(
        showscale=True,
        colorscale="Viridis",
        size=20,
        color=node_color,
        line_width=2
    )
)

# ----------------------------
# 5. Build Interactive Graph
# ----------------------------
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title="Query Clustering with Louvain",
                    titlefont_size=20,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[dict(
                        text="Queries clustered by claim overlap",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002
                    )],
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False)
                ))

fig.show()
