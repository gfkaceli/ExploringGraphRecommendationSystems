import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.layout import bipartite_layout
from networkx.algorithms import bipartite
import pandas as pd
import random


def visualize_bipartite(dataset_path, n_samples=200, n_users=10, save_path="visualization/graph.png"):
    """
    Visualizes a subgraph of the MovieLens dataset.

    Parameters:
    - dataset_path: str, path to the ratings.csv file.
    - n_samples: int, number of samples to visualize from the dataset.
    - n_users: int, number of users to select for the subgraph visualization.
    - save_path: str, path to save the visualization image.
    """
    df = pd.read_csv(dataset_path)
    # Sample a subset of the dataset to keep the graph manageable
    df = df.sample(n=n_samples)

    user = df['UID']
    movies = df['MID']

    g = nx.Graph()

    g.add_nodes_from(user, bipartite=0)
    g.add_nodes_from(movies, bipartite=1)

    # Add edges with ratings as weights
    for _, row in df.iterrows():
        g.add_edge(row['UID'], row['MID'], weight=row['rating'])

    selected_users = random.sample(list(df['UID'].unique()), n_users)

    # Select movies rated by these users
    selected_movies = df[df['UID'].isin(selected_users)]['MID'].unique()

    # Create a subgraph with selected users and movies
    subgraph_nodes = list(selected_users) + list(selected_movies)
    h = g.subgraph(subgraph_nodes)

    # Position nodes using the bipartite layout
    pos = nx.bipartite_layout(h, selected_users)

    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw(h, pos, with_labels=True, node_color=['skyblue' if node in selected_users else 'lightgreen' for node in h],
            edge_color='gray', font_size=8, node_size=50)

    # Draw edge labels to show ratings
    edge_labels = nx.get_edge_attributes(h, 'weight')
    nx.draw_networkx_edge_labels(h, pos, edge_labels=edge_labels, font_size=7)

    plt.title("Subgraph of the MovieLens Dataset")
    plt.axis('off')
    plt.savefig(save_path)
    plt.clf()


""" from this we can see the bipartite nature of the graph"""

for i in range(5): # here generate 5 samples of the bipartite graph
    visualize_bipartite("datasets/encoded/ratings.csv", 200, 10, f'visualization/graph{i}')






