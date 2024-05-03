import networkx as nx
import matplotlib.pyplot as plt

# Market conditions timesteps and risk profile ranges derived empirically
conditions_data = {
    9: {"timesteps": 226, "range": (0.01, 0.17)},
    5: {"timesteps": 486, "range": (0.01, 0.17)},
    8: {"timesteps": 606, "range": (0.01, 0.17)},
    4: {"timesteps": 251, "range": (0.01, 0.19)},
    12: {"timesteps": 156, "range": (0.01, 0.21)},
    1: {"timesteps": 55, "range": (0.01, 0.27)},
    13: {"timesteps": 363, "range": (0.01, 0.29)},
    6: {"timesteps": 156, "range": (0.01, 0.4)},
    2: {"timesteps": 83, "range": (0.01, 0.4)},
    3: {"timesteps": 12, "range": (0.01, 0.4)},
    7: {"timesteps": 43, "range": (0.01, 0.41)},
    11: {"timesteps": 71, "range": (0.01, 0.47)},
    15: {"timesteps": 134, "range": (0.01, 0.56)},
    14: {"timesteps": 61, "range": (0.01, 0.8)},
    10: {"timesteps": 41, "range": (0.01, 0.91)}
}

# Merging information derived after visualising first time
merging_info = {
    0: 6, 2: 6, 3: 6, 4: 8, 5: 8, 7: 6, 9: 8, 11: 6, 12: 8, 1: 13
}

# Initialize the graph
G = nx.DiGraph()

# Add nodes with sizes proportional to timesteps and colors based on range
for cond, info in conditions_data.items():
    color = plt.cm.viridis((info['range'][1] - 0.01) / 0.9)
    G.add_node(cond, size=info['timesteps'], color=color)

# Add edges based on merging information
for src, dest in merging_info.items():
    if src in conditions_data and dest in conditions_data:
        G.add_edge(src, dest)

# Setup the figure and axis
fig, ax = plt.subplots(figsize=(12, 8))
pos = nx.kamada_kawai_layout(G) 

node_sizes = [G.nodes[node]['size']*10 for node in G]
node_colors = [G.nodes[node]['color'] for node in G]

nx.draw(G, pos, node_size=node_sizes, node_color=node_colors, with_labels=True, cmap=plt.cm.viridis, arrows=True, arrowstyle='-|>', arrowsize=10, ax=ax)
plt.title('Market Condition Merging based on Neighbor Clustering')

# Create a colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0.01, vmax=0.91))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
cbar.set_label('Risk Profile Upper Bound')

plt.show()