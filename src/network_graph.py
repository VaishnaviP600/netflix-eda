"""
src/network_graph.py
---------------------
Actor and Director collaboration network analysis using NetworkX.

WHAT IS A NETWORK GRAPH?
  - Nodes  = people (actors / directors)
  - Edges  = two people appeared in the same title together
  - Edge weight = how many titles they shared
  - Degree = how many connections a node has (= how prolific the person is)

WHAT WE CAN DISCOVER:
  - Which actors appear in the most Netflix titles? (highest degree)
  - Which actors frequently co-star? (high edge weight)
  - Are there clusters of actors who always work together?
  - Which directors collaborate with the most actors?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import Counter
from itertools import combinations

import networkx as nx
from src.data_cleaner import explode_cast

NETFLIX_RED = "#E50914"


# ══════════════════════════════════════════════════════════════════════════════
# 1. BUILD GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_actor_graph(df: pd.DataFrame,
                      min_titles: int = 5,
                      max_cast_per_title: int = 6) -> nx.Graph:
    """
    Build an undirected weighted graph of actor co-occurrences.

    Parameters:
      min_titles         : only include actors who appear in >= N titles
      max_cast_per_title : cap cast list length to avoid explosion of edges

    Algorithm:
      For each title with a known cast:
        For each PAIR of actors in that cast:
          Add an edge between them (or increment edge weight if it already exists)

    Result: G.nodes = actors, G.edges = co-appearances,
            G[a][b]["weight"] = number of shared titles
    """
    # Count appearances per actor
    df_cast = explode_cast(df[df["cast"] != "Unknown"])
    actor_counts = df_cast["actor"].str.strip().value_counts()
    prolific_actors = set(actor_counts[actor_counts >= min_titles].index)

    G = nx.Graph()

    for _, row in df.iterrows():
        if row["cast"] == "Unknown" or not isinstance(row["cast"], str):
            continue
        cast_list = [a.strip() for a in row["cast"].split(",")][:max_cast_per_title]
        # Keep only prolific actors
        cast_list = [a for a in cast_list if a in prolific_actors]

        for a1, a2 in combinations(cast_list, 2):
            if G.has_edge(a1, a2):
                G[a1][a2]["weight"] += 1
            else:
                G.add_edge(a1, a2, weight=1)

    # Add node attribute: total appearances
    nx.set_node_attributes(G, {
        actor: int(actor_counts.get(actor, 0))
        for actor in G.nodes()
    }, name="appearances")

    print(f"Actor graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def build_director_actor_graph(df: pd.DataFrame, min_titles: int = 3) -> nx.Graph:
    """
    Bipartite graph: directors ↔ actors.
    An edge means the director worked with that actor.
    Node types: "director" or "actor".
    """
    df_known = df[(df["director"] != "Unknown") & (df["cast"] != "Unknown")]

    dir_counts = df_known["director"].value_counts()
    prolific_dirs = set(dir_counts[dir_counts >= min_titles].index.str.strip())

    G = nx.Graph()

    for _, row in df_known.iterrows():
        directors = [d.strip() for d in str(row["director"]).split(",")]
        actors    = [a.strip() for a in str(row["cast"]).split(",")][:5]

        for director in directors:
            if director not in prolific_dirs:
                continue
            G.add_node(director, node_type="director", label=director)
            for actor in actors:
                G.add_node(actor, node_type="actor", label=actor)
                if G.has_edge(director, actor):
                    G[director][actor]["weight"] += 1
                else:
                    G.add_edge(director, actor, weight=1)

    print(f"Director-Actor graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ══════════════════════════════════════════════════════════════════════════════
# 2. GRAPH STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def top_nodes_by_degree(G: nx.Graph, n: int = 20) -> pd.DataFrame:
    """Return top N most connected actors (degree = number of co-stars)."""
    degree = dict(G.degree(weight="weight"))
    appearances = nx.get_node_attributes(G, "appearances")
    df = pd.DataFrame({
        "actor"       : list(degree.keys()),
        "connections" : list(degree.values()),
        "appearances" : [appearances.get(a, 0) for a in degree.keys()],
    }).sort_values("connections", ascending=False).head(n)
    return df.reset_index(drop=True)


def top_collaborations(G: nx.Graph, n: int = 20) -> pd.DataFrame:
    """Return top N actor pairs with highest co-appearance count."""
    edges = [(u, v, d["weight"]) for u, v, d in G.edges(data=True)]
    df = pd.DataFrame(edges, columns=["actor_1", "actor_2", "shared_titles"])
    return df.sort_values("shared_titles", ascending=False).head(n).reset_index(drop=True)


def get_network_stats(G: nx.Graph) -> dict:
    """Summary statistics for the graph."""
    return {
        "nodes"            : G.number_of_nodes(),
        "edges"            : G.number_of_edges(),
        "avg_degree"       : round(sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1), 2),
        "density"          : round(nx.density(G), 6),
        "connected_comps"  : nx.number_connected_components(G),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_top_actors_bar(df_top: pd.DataFrame) -> plt.Figure:
    """Horizontal bar: top N actors by number of Netflix titles."""
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [NETFLIX_RED if i == 0 else "#333" for i in range(len(df_top))]
    bars = ax.barh(df_top["actor"][::-1], df_top["appearances"][::-1],
                   color=colors[::-1], edgecolor="none")
    for bar, val in zip(bars, df_top["appearances"][::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(val), va="center", color="#ccc", fontsize=10)
    ax.set_title("Most Prolific Actors on Netflix", fontsize=14, color="#fff")
    ax.set_xlabel("Number of Titles", color="#aaa")
    ax.grid(axis="x", alpha=0.2)
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#1a1a1a")
    fig.tight_layout()
    return fig


def plot_network(G: nx.Graph, title: str = "Actor Co-occurrence Network",
                 max_nodes: int = 80) -> plt.Figure:
    """
    Draw the network graph.

    Layout: spring layout (nodes repel each other; edges attract).
    Node size = number of appearances (bigger = more titles).
    Edge width = number of shared titles (thicker = closer collaboration).
    Node colour = degree (number of connections) — darker = more connected.
    """
    # Subsample to most connected nodes for readability
    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
        G = G.subgraph([n for n, _ in top_nodes]).copy()

    appearances = nx.get_node_attributes(G, "appearances")
    degrees     = dict(G.degree())

    node_sizes  = [max(50, appearances.get(n, 1) * 15) for n in G.nodes()]
    edge_widths = [G[u][v].get("weight", 1) * 0.5 for u, v in G.edges()]
    node_colors = [degrees.get(n, 1) for n in G.nodes()]

    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#141414")
    ax.set_facecolor("#141414")

    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=edge_widths, alpha=0.25, edge_color="#555")
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax,
                                    node_size=node_sizes,
                                    node_color=node_colors,
                                    cmap=plt.cm.Reds, alpha=0.9)
    # Labels only for top 20 most connected nodes
    top20 = [n for n, _ in sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]]
    label_pos = {n: pos[n] for n in top20 if n in pos}
    nx.draw_networkx_labels(G, label_pos, ax=ax,
                             labels={n: n for n in top20},
                             font_size=7, font_color="#fff")
    plt.colorbar(nodes, ax=ax, label="Degree (connections)", shrink=0.6)
    ax.set_title(title, fontsize=15, color="#fff", pad=16)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_collaboration_heatmap(df_collab: pd.DataFrame, n: int = 15) -> plt.Figure:
    """Heatmap of actor collaboration frequency."""
    import seaborn as sns

    actors = list(set(df_collab["actor_1"].tolist() + df_collab["actor_2"].tolist()))[:n]
    matrix = pd.DataFrame(0, index=actors, columns=actors)
    for _, row in df_collab.iterrows():
        a1, a2 = row["actor_1"], row["actor_2"]
        if a1 in matrix.index and a2 in matrix.columns:
            matrix.loc[a1, a2] = row["shared_titles"]
            matrix.loc[a2, a1] = row["shared_titles"]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(matrix, ax=ax, cmap="Reds", linewidths=0.3,
                linecolor="#222", annot=True, fmt="d")
    ax.set_title("Actor Collaboration Matrix (shared titles)", fontsize=13, color="#fff")
    fig.patch.set_facecolor("#1a1a1a")
    fig.tight_layout()
    return fig
