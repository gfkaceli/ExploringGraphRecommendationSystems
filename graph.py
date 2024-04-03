import networkx as nx
import matplotlib.pyplot as plt

g = nx.Graph()

g.add_node(1)
g.add_node(2)
g.add_node(3)
g.add_nodes_from([4, 5, 6])

g.add_edge(1, 2)
g.add_edge(4, 2)
g.add_edge(1, 6)
g.add_edge(3, 5)
g.add_edge(2, 3)
g.add_edge(5, 1)
g.add_edge(3, 4)

nx.draw(g, with_labels=True, node_color='lightblue', edge_color='gray')
plt.show()

# Basic analysis
print("Number of nodes:", g.number_of_nodes())
print("Number of edges:", g.number_of_edges())
print("List of all nodes:", list(g.nodes))
print("List of all edges:", list(g.edges))
print("Neighbors of node 2:", list(g.neighbors(2)))

# Network metrics
print("Degree of each node:", dict(g.degree()))
print("Clustering coefficient of the network:", nx.clustering(g))





