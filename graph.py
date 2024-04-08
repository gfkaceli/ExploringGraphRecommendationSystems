import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.layout import bipartite_layout
from networkx.algorithms import bipartite
import pandas as pd
import random

df = pd.read_csv("datasets/encoded/ratings.csv")
# Sample a subset of the dataset to keep the graph small
df = df.sample(n=200)

user = df['UID']
movies = df['MID']

g = nx.Graph()

g.add_nodes_from(user, bipartite=0)
g.add_nodes_from(movies, bipartite=1)

# Add edges with ratings as weights
for _, row in df.iterrows():
    g.add_edge(row['UID'], row['MID'], weight=row['rating'])

selected_users = random.sample(list(df['UID'].unique()), 10)

# Select movies rated by these users
selected_movies = df[df['UID'].isin(selected_users)]['MID'].unique()

# Create a subgraph with selected users and movies
subgraph_nodes = list(selected_users) + list(selected_movies)
h = g.subgraph(subgraph_nodes)

# Position nodes using the bipartite layout
pos = nx.bipartite_layout(h, selected_users)

# Draw the graph
plt.figure(figsize=(12, 8))
# blue corresponds to the user id, green will correspond to the movie id and the number in between is the weight
nx.draw(h, pos, with_labels=True, node_color=['skyblue' if node in selected_users else 'lightgreen' for node in h],
        edge_color='gray', font_size=8, node_size=50)

# Draw edge labels to show ratings
edge_labels = nx.get_edge_attributes(h, 'weight')
nx.draw_networkx_edge_labels(h, pos, edge_labels=edge_labels, font_size=7)

plt.title("Subgraph of the MovieLens Dataset")
plt.axis('off')
plt.savefig("visualization/graph")
plt.clf()

""" from this we can see the bipartite nature of the graph"""




