import networkx as nx
import matplotlib.pyplot as plt

# Create a directed acyclic graph (DAG)
graph = nx.DiGraph()

# Add nodes
graph.add_nodes_from(["SES", "Exercise", "Diet", "WeightLoss"])

# Add edges
graph.add_edges_from([("SES", "Exercise"),
                      ("SES", "Diet"),
                      ("Exercise", "WeightLoss"),
                      ("Diet", "WeightLoss")])

# Draw the graph
nx.draw(graph, with_labels=True)
plt.show()