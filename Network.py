import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

# Create the graph
G = nx.Graph()

# Define the edges with connections
edges = [
    (1, 2), (1, 5), (1, 7), (1, 6), (1, 3), (1, 4),  # Node 1 connections
    (2, 5), (2, 6),  # Node 2 connections
    (3, 6), (3, 8),  # Node 3 connections
    (4, 8),          # Node 4 connections
    (5, 7),          # Node 5 connections
    (6, 2), (6, 3),  # Node 6 connections
    (7, 5),          # Node 7 connections
    (8, 3), (8, 4)   # Node 8 connections
]

# Add edges to the graph
G.add_edges_from(edges)

# Assign random failure probabilities to each edge (e.g., between 0.1 and 0.3)
np.random.seed(42)  # For reproducibility
for edge in G.edges:
    G.edges[edge]["failure_prob"] = round(np.random.uniform(0.1, 0.3), 2)

# Visualize the graph with edge failure probabilities
def visualize_graph_with_probabilities(G):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 8))

    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=700, font_size=12, font_weight="bold")

    # Add edge labels (failure probabilities)
    edge_labels = nx.get_edge_attributes(G, "failure_prob")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    plt.title("Graph with Edge Failure Probabilities", fontsize=14)
    plt.show()

visualize_graph_with_probabilities(G)

# **********************************************************************************
# *****************************Analytical Solution**********************************
# **********************************************************************************

# Create the probability matrix P based on failure probabilities
nodes = list(G.nodes)
node_to_index = {node: idx for idx, node in enumerate(nodes)}  # Map node labels to matrix indices
n = len(nodes)
P = np.zeros((n, n))

# Populate the probability matrix
for i, node1 in enumerate(nodes):
    for j, node2 in enumerate(nodes):
        if G.has_edge(node1, node2):
            P[i, j] = 1 - G.edges[node1, node2]["failure_prob"]  # Success probability

# Compute the connectivity matrix: (I - P)^(-1)
I = np.eye(n)  # Identity matrix
try:
    connectivity_matrix = np.linalg.inv(I - P)
except np.linalg.LinAlgError:
    raise ValueError("Matrix (I - P) is singular and cannot be inverted.")
print (P)
print(connectivity_matrix)
# ***********************************************************************************
# *********************************Numerical Solution********************************
# ***********************************************************************************

# Monte Carlo Simulation (Numerical Solution)
def monte_carlo_simulation(G, num_trials=1000):
    num_nodes = len(G.nodes)
    success_count = np.zeros((num_nodes, num_nodes))

    # Run Monte Carlo simulation
    for trial in range(num_trials):
        # Create a fresh copy of the graph for each trial to avoid permanent modifications
        G_trial = G.copy()

        # For each edge, decide whether it exists based on the failure probability
        for edge in G_trial.edges:
            node1, node2 = edge
            failure_prob = G_trial.edges[edge]["failure_prob"]
            if random.random() > failure_prob:  # Edge exists
                G_trial[edge[0]][edge[1]]["exists"] = True
            else:  # Edge fails
                G_trial[edge[0]][edge[1]]["exists"] = False
        
        # Count the number of successful connections between nodes
        for i in range(num_nodes):
            for j in range(num_nodes):
                if nx.has_path(G_trial, nodes[i], nodes[j]):  # There is a path between node i and node j
                    success_count[i, j] += 1

    # Compute the connectivity probabilities
    connectivity_probabilities = success_count / num_trials
    return connectivity_probabilities

# Perform the Monte Carlo simulation
num_trials = 1000
monte_carlo_probs = monte_carlo_simulation(G, num_trials)

# Bar graph to compare the results
# plt.figure(figsize=(10, 6))

# Create bar plots for each node pair
for i in range(n):
    for j in range(i+1, n):
        node_pair = f"({nodes[i]}, {nodes[j]})"
        analytic_value = connectivity_matrix[i, j]
        monte_carlo_value = monte_carlo_probs[i, j]
        
        plt.bar(node_pair, analytic_value, color='b', width=0.4, label="Analytical" if i == 0 else "")
        plt.bar(node_pair, monte_carlo_value, color='r', width=0.4, label="Monte Carlo" if i == 0 else "")

plt.xticks(rotation=90)
plt.xlabel("Node Pair", fontsize=12)
plt.ylabel("Probability", fontsize=12)
plt.title("Comparison of Analytical and Monte Carlo Connectivity Probabilities", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()