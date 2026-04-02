# MIT License
#
# Copyright (c) 2024 Durai Rajamanickam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Creating a Simple Causal Graph ---
def create_simple_graph():
    """Creates and visualizes a simple causal graph: Exercise -> Weight Loss."""

    graph = nx.DiGraph()  # Directed Acyclic Graph
    graph.add_edges_from([('Exercise', 'Weight Loss')])

    plt.figure(figsize=(8, 6))
    nx.draw_circular(graph, with_labels=True, node_size=2000, node_color='skyblue', font_size=12, arrowsize=20)  # Adjust layout for clarity
    plt.title('Simple Causal Graph')
    plt.show()
    return graph


# --- 2. Creating a More Complex Graph ---
def create_complex_graph():
    """Creates and visualizes a more complex causal graph with confounding and mediation."""

    graph = nx.DiGraph()
    graph.add_edges_from([('SES', 'Exercise'), ('SES', 'Diet'),
                          ('Exercise', 'Weight Loss'), ('Diet', 'Weight Loss')])

    plt.figure(figsize=(8, 6))
    nx.draw_circular(graph, with_labels=True, node_size=2000, node_color='lightgreen', font_size=12, arrowsize=20)
    plt.title('Complex Causal Graph')
    plt.show()
    return graph


# --- 3. Representing a Graph as an Adjacency Matrix ---
def graph_to_adjacency_matrix(graph):
    """Converts a networkx graph to an adjacency matrix."""

    nodes = list(graph.nodes)
    num_nodes = len(nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    node_to_index = {node: i for i, node in enumerate(nodes)}  # Map node to matrix index

    for i in range(num_nodes):
        for j in range(num_nodes):
            if graph.has_edge(nodes[i], nodes[j]):
                adj_matrix[i, j] = 1

    print("\n--- Adjacency Matrix ---")
    print("Nodes:", nodes)
    print(adj_matrix)
    return adj_matrix, nodes


# --- 4. Demonstrating a Chain Structure ---
def demonstrate_chain():
    """Illustrates a chain structure and path blocking."""

    graph_chain = nx.DiGraph()
    graph_chain.add_edges_from([('A', 'B'), ('B', 'C')])

    plt.figure(figsize=(6, 4))
    nx.draw_circular(graph_chain, with_labels=True, node_size=2000, node_color='lightcoral', font_size=12, arrowsize=20)
    plt.title('Chain Structure (A -> B -> C)')
    plt.show()

    print("\n--- Chain Structure ---")
    print("In a chain, B mediates the effect of A on C.")
    print("Conditioning on B blocks the path from A to C.")


# --- 5. Demonstrating a Fork Structure ---
def demonstrate_fork():
    """Illustrates a fork structure and confounding."""

    graph_fork = nx.DiGraph()
    graph_fork.add_edges_from([('B', 'A'), ('B', 'C')])

    plt.figure(figsize=(6, 4))
    nx.draw_circular(graph_fork, with_labels=True, node_size=2000, node_color='gold', font_size=12, arrowsize=20)
    plt.title('Fork Structure (Confounding)')
    plt.show()

    print("\n--- Fork Structure ---")
    print("In a fork, B is a common cause of A and C (confounder).")
    print("Not controlling for B can create a spurious association between A and C.")


# --- 6. Demonstrating a Collider Structure ---
def demonstrate_collider():
    """Illustrates a collider structure and collider bias."""

    graph_collider = nx.DiGraph()
    graph_collider.add_edges_from([('A', 'B'), ('C', 'B')])

    plt.figure(figsize=(6, 4))
    nx.draw_circular(graph_collider, with_labels=True, node_size=2000, node_color='lightblue', font_size=12, arrowsize=20)
    plt.title('Collider Structure')
    plt.show()

    print("\n--- Collider Structure ---")
    print("In a collider, B is caused by both A and C.")
    print("Conditioning on B can open a non-causal path between A and C, creating bias.")


# --- 7. Main Execution ---
if __name__ == '__main__':
    simple_graph = create_simple_graph()
    complex_graph = create_complex_graph()

    adj_matrix, nodes = graph_to_adjacency_matrix(complex_graph)

    demonstrate_chain()
    demonstrate_fork()
    demonstrate_collider()

