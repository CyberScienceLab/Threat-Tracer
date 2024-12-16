
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import networkx as nx
from sklearn.cluster import SpectralClustering

df = pd.read_csv('Clean_dataset.csv')

# Define Nodes:
df['node_id'] = range(len(df))  # Unique identifier for each command instance

# Define Edges:
edges = []
for i in range(len(df) - 1):
    edge = (df['node_id'][i], df['node_id'][i + 1])
    edges.append(edge)
edges = torch.tensor(edges, dtype=torch.long).t().contiguous()

G_behavior = nx.DiGraph()

# Add nodes
for node_id in df['node_id']:
    G_behavior.add_node(node_id)

# Add edges
for edge in edges.t().numpy():
    G_behavior.add_edge(edge[0], edge[1])

# Community detection
import leidenalg as la
import igraph as ig
import time
# Start measuring time
start_time = time.time()

# Function to convert NetworkX graph to iGraph
def convert_networkx_to_igraph(nx_graph):
    g = ig.Graph(directed=True)
    g.add_vertices(list(nx_graph.nodes()))
    g.add_edges(list(nx_graph.edges()))
    return g

# Convert the NetworkX graph to an iGraph graph
G_behavior_ig = convert_networkx_to_igraph(G_behavior)

# Apply the Leiden algorithm
partition = la.find_partition(G_behavior_ig, la.ModularityVertexPartition)

# Map the partition back to the NetworkX nodes
community_map = {}
for idx, community in enumerate(partition):
    for node_id in community:
        node = G_behavior_ig.vs[node_id]['name']
        community_map[node] = idx

# Create a new graph for the summarized version
G_summary = nx.DiGraph()
# # Create a new graph for the summarized version
# G_summary = nx.DiGraph()

# Initialize community information storage
community_info = {i: {
    'nodes': set(),
    'edges': set(),
    'commands': set(),
    'statuses': set(),
    'operations': set(),
    'tactics': set(),
    'techniques': set(),
    'pids': set(),
    'platforms': set(),
    'executors': set(),
    'abilities': set()
} for i in set(community_map.values())}

# Plural forms of node types
plural_map = {
    'operation': 'operations',
    'tactic': 'tactics',
    'technique': 'techniques',
    'log': 'logs',
    'pid': 'pids',
    'platform': 'platforms',
    'executor': 'executors',
    'ability': 'abilities'
}

# Assign nodes to communities and aggregate features
for node, community in community_map.items():
    community_info[community]['nodes'].add(node)
    node_data = G_behavior.nodes[node]

    # Check if the node is a log and aggregate command and status
    if node_data.get('type') == 'log':
        community_info[community]['commands'].add(node_data.get('command', ''))
        community_info[community]['statuses'].add(node_data.get('status', ''))
    else:
        feature_type = node_data.get('type')
        if feature_type:
            plural_feature_type = plural_map.get(feature_type, feature_type + 's')
            community_info[community][plural_feature_type].add(node)

# Add supernodes to the summary graph and assign aggregated features
for community, info in community_info.items():
    supernode_id = f"supernode_{community}"
    G_summary.add_node(supernode_id, **{feature: list(values) for feature, values in info.items() if feature != 'nodes'})

# Connect supernodes based on inter-community edges
for node in G_behavior:
    node_community = community_map[node]
    for neighbor in G_behavior.neighbors(node):
        neighbor_community = community_map[neighbor]
        if node_community != neighbor_community:
            supernode_source = f"supernode_{node_community}"
            supernode_target = f"supernode_{neighbor_community}"
            G_summary.add_edge(supernode_source, supernode_target)

# Stop measuring time
end_time = time.time()

# Calculate the time taken to create the graph
time_taken = end_time - start_time
print(f"Time taken to create the graph: {time_taken:.2f} seconds")

G_summary

import matplotlib.pyplot as plt

# Choose a layout for the graph
pos = nx.spring_layout(G_summary, k=0.5, iterations=20)

# Draw the graph
nx.draw(G_summary, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray', font_size=10, font_weight='bold')

# Optionally, add labels to the supernodes
for node, data in G_summary.nodes(data=True):
    label = f"{node}\nOps: {len(data.get('operations', []))}\nTechs: {len(data.get('techniques', []))}"
    plt.text(pos[node][0], pos[node][1], label, fontsize=9, ha='center', va='center')


# Save the figure with a high DPI
plt.savefig('MITRE summarization.png', dpi=400, bbox_inches='tight')

# Show the plot
# plt.title("Summarized Graph Visualization")
plt.show()



import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, jaccard_score
import leidenalg as la

def analyze_community_count_variation(graph, base_partition, k_values):
    """
    Vary the number of communities and calculate ARI score between
    each partition and the base partition.
    """
    scores = {}
    for k in k_values:
        # Apply Spectral Clustering with different numbers of communities
        clustering = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=0)
        community_labels = clustering.fit_predict(nx.adjacency_matrix(graph))

        # Calculate ARI score with the base partition, using a fallback for missing nodes
        base_labels = [base_partition.get(node, -1) for node in graph.nodes()]  # -1 as a default label for missing nodes
        scores[k] = adjusted_rand_score(base_labels, community_labels)
        print(f"ARI score with k={k}: {scores[k]}")
    return scores
# def threshold_based_community_analysis(graph, initial_partition, weight_thresholds):
#     """
#     Vary edge weight thresholds and calculate Jaccard similarity
#     for community assignments at each threshold.
#     """
#     for threshold in weight_thresholds:
#         # Remove edges below the threshold
#         perturbed_graph = graph.copy()
#         edges_to_remove = [(u, v) for u, v, w in perturbed_graph.edges(data='weight') if w < threshold]
#         perturbed_graph.remove_edges_from(edges_to_remove)

#         # Apply community detection on the modified graph
#         perturbed_ig_graph = convert_networkx_to_igraph(perturbed_graph)
#         perturbed_partition = la.find_partition(perturbed_ig_graph, la.ModularityVertexPartition)

#         # Map results back to NetworkX for comparison
#         perturbed_map = {}
#         for idx, community in enumerate(perturbed_partition):
#             for node_id in community:
#                 node = perturbed_ig_graph.vs[node_id]['name']
#                 perturbed_map[node] = idx

#         # Calculate Jaccard similarity with the initial partition
#         initial_labels = [initial_partition[node] for node in perturbed_graph.nodes()]
#         perturbed_labels = [perturbed_map.get(node, -1) for node in perturbed_graph.nodes()]
#         jaccard_sim = jaccard_score(initial_labels, perturbed_labels, average='weighted')
#         print(f"Jaccard Similarity at threshold {threshold}: {jaccard_sim}")
def convert_networkx_to_igraph(nx_graph):
    # Step 1: Map NetworkX nodes to consecutive integers for iGraph compatibility
    node_mapping = {node: idx for idx, node in enumerate(nx_graph.nodes())}

    # Step 2: Create iGraph object with remapped nodes and edges
    g = ig.Graph(directed=True)
    g.add_vertices(len(node_mapping))  # Add all nodes with new integer IDs
    g.add_edges([(node_mapping[u], node_mapping[v]) for u, v in nx_graph.edges() if u in node_mapping and v in node_mapping])

    return g, node_mapping  # Return the graph and the node mapping

# Apply in threshold_based_community_analysis
def threshold_based_community_analysis(graph, initial_partition, weight_thresholds):
    for threshold in weight_thresholds:
        # Filter edges by threshold
        perturbed_graph = graph.copy()
        edges_to_remove = [(u, v) for u, v, w in perturbed_graph.edges(data='weight', default=1.0) if w < threshold]
        perturbed_graph.remove_edges_from(edges_to_remove)

        # Convert NetworkX to iGraph with node mapping
        perturbed_ig_graph, node_mapping = convert_networkx_to_igraph(perturbed_graph)

        # Apply Leiden algorithm on the perturbed iGraph
        perturbed_partition = la.find_partition(perturbed_ig_graph, la.ModularityVertexPartition)

        # Map partition results back to NetworkX nodes
        perturbed_map = {node: idx for idx, community in enumerate(perturbed_partition) for node in community}

        # Calculate Jaccard similarity with the initial partition
        initial_labels = [initial_partition.get(node, -1) for node in perturbed_graph.nodes()]
        perturbed_labels = [perturbed_map.get(node, -1) for node in perturbed_graph.nodes()]
        jaccard_sim = jaccard_score(initial_labels, perturbed_labels, average='weighted')
        print(f"Jaccard Similarity at threshold {threshold}: {jaccard_sim}")

def modularity_resolution_analysis(igraph_graph, initial_partition, resolutions):
    """
    Adjust modularity resolution and calculate community sizes
    for each resolution.
    """
    for res in resolutions:
        partition = la.find_partition(igraph_graph, la.RBConfigurationVertexPartition, resolution_parameter=res)
        community_sizes = [len(community) for community in partition]
        print(f"Resolution {res} results in {len(partition)} communities with sizes: {community_sizes}")

# Example usage
# k_values = [5, 10, 50, 100, 130, 150]  # Example range for community counts
k_values = range(2, 10)
weight_thresholds = np.linspace(0.1, 0.9, 9)  # Range for weight thresholding
resolutions = np.arange(0.1, 2.0, 0.2)  # Range for modularity resolution parameter

# Run the analyses
print("Analyzing community count variation:")
analyze_community_count_variation(G_behavior, community_map, k_values)

print("\nAnalyzing threshold-based community transitions:")
threshold_based_community_analysis(G_behavior, community_map, weight_thresholds)

print("\nAnalyzing modularity resolution sensitivity:")
modularity_resolution_analysis(G_behavior_ig, community_map, resolutions)

# Count supernodes and superedges
num_supernodes = G_summary.number_of_nodes()
num_superedges = G_summary.number_of_edges()

print(f"Number of supernodes: {num_supernodes}")
print(f"Number of superedges: {num_superedges}")

# Extract Node Features:
selected_features = [
    'finished_timestamp',
    'collected_timestamp', 'delegated_timestamp', 'operation_metadata.operation_start',
    'agent_reported_time', 'agent_metadata.created', 'pid',
    'agent_metadata.ppid', 'agent_metadata.pid', "attack_metadata.tactic_encoded", "attack_metadata.technique_name_encoded"
]
node_features = df[selected_features]
# node_features = torch.tensor(node_features.values, dtype=torch.float)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the features
scaled_features = scaler.fit_transform(node_features)

# Convert the scaled features to a PyTorch tensor
node_features = torch.tensor(scaled_features, dtype=torch.float)

# Encode Labels:
labels = LabelEncoder().fit_transform(df['operation_metadata.operation_adversary'])
labels = torch.tensor(labels, dtype=torch.long)

# # Initialize a dictionary to store aggregated features for each community
# community_features = {}

# for idx, community in enumerate(partition):
#     # Extract node features for nodes in this community
#     community_node_features = [node_features[node] for node in community]

#     # Check if the community is not empty
#     if community_node_features:
#         # Convert list of features to numpy array for proper mean calculation
#         features_array = np.array(community_node_features)
#         # Aggregate features using mean
#         community_features[idx] = np.mean(features_array, axis=0)

# # Now, community_features[idx] contains the aggregated feature vector for community idx



community_features = {}

for idx, community in enumerate(partition):
    # Extract node features for nodes in this community
    community_node_features = [node_features[node].numpy() for node in community if node < len(node_features)]

    # Ensure community has nodes and the feature dimensions are consistent
    if community_node_features:
        # Convert list of numpy arrays to a numpy array
        features_array = np.stack(community_node_features)
        # Aggregate features using mean
        community_features[idx] = np.mean(features_array, axis=0)

# Assuming community_features is a dictionary or similar structure
community_features_array = np.array([community_features[i] for i in range(len(community_features))])

# Scale the community features
scaled_community_features = scaler.fit_transform(community_features_array)

# Convert the scaled features to a PyTorch tensor
community_features_tensor = torch.tensor(scaled_community_features, dtype=torch.float)

print(list(G_summary.edges())[:5])  # Print the first 5 edges to inspect their format

# Extract unique node names from the edges
unique_nodes = set()
for edge in G_summary.edges():
    unique_nodes.update(edge)

# Create a mapping from node names to integers
node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}

# Print the mapping to verify
print(node_mapping)

# Convert edges to integer format using the mapping
integer_edges = [(node_mapping[edge[0]], node_mapping[edge[1]]) for edge in G_summary.edges()]

# Print the first few converted edges to verify
print(integer_edges[:5])

edge_index = torch.tensor(integer_edges, dtype=torch.long).t().contiguous()

# Convert to tensor
# node_features_summarized = torch.tensor([community_features[i] for i in range(len(community_features))], dtype=torch.float)
# Convert the list of NumPy arrays to a single NumPy array
node_features_np = np.array([community_features[i] for i in range(len(community_features))])

# Then, convert this NumPy array to a PyTorch tensor
node_features_summarized = torch.tensor(node_features_np, dtype=torch.float)

# Encode Labels:
# labels = LabelEncoder().fit_transform(df['operation_metadata.operation_adversary'])
# labels = torch.tensor(labels, dtype=torch.long)

from collections import Counter

# Assuming 'labels' is your tensor of original node labels
node_labels = labels.numpy()

# Assign a label to each community
community_labels = []
for community in partition:
    community_label_counts = Counter(node_labels[node] for node in community)
    most_common_label = community_label_counts.most_common(1)[0][0]
    community_labels.append(most_common_label)

# Convert community_labels to a tensor
community_labels_tensor = torch.tensor(community_labels, dtype=torch.long)

# Create a PyTorch Geometric data object
# data_summarized = Data(x=node_features_summarized, edge_index=edge_index,y=labels)
data_summarized = Data(x=community_features_tensor, edge_index=edge_index, y=community_labels_tensor)  # If using node features

class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, 32, heads=8, concat=True, dropout=0.7)
        self.conv2 = GATConv(32* 8, 16, heads=8, concat=True, dropout=0.7)
        self.conv3 = GATConv(16 * 8, 8, heads=8, concat=True, dropout=0.7)
        self.conv4 = GATConv(8 * 8, num_classes, heads=1, concat=False, dropout=0.7)
        self.skip_connection = torch.nn.Linear(num_node_features, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.7, training=self.training)

        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.7, training=self.training)

        x3 = self.conv3(x2, edge_index)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, p=0.7, training=self.training)

        x4 = self.conv4(x3, edge_index)

        # Add a skip connection
        skip = self.skip_connection(x)

        return F.log_softmax(x4 + skip, dim=1)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def l1_penalty(model):
    l1_loss = 0.0
    for param in model.parameters():
        l1_loss += param.abs().sum()
    return l1_loss

l1_lambda = 0.005

from sklearn.utils.class_weight import compute_class_weight

# # Assuming 'labels' is your tensor of labels
class_weights = compute_class_weight('balanced', classes=np.unique(labels.numpy()), y=labels.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Instantiate the model
from torch_geometric.nn import GATConv

num_classes = len(df['operation_metadata.operation_adversary'].unique())
model = GAT(num_node_features=community_features_tensor.shape[1], num_classes=num_classes)

# Define loss function and optimizer
# criterion = torch.nn.CrossEntropyLoss()

# Apply to your loss function
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Add L2 regularization here

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, gamma=3, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

import numpy as np

# Total number of nodes
num_nodes = data_summarized.num_nodes

# Define the split proportions
train_proportion = 0.8
val_proportion = 0.2

# Create indices for all nodes
indices = np.arange(num_nodes)

# Shuffle the indices
np.random.shuffle(indices)

# Calculate the number of training and validation nodes
num_train = int(train_proportion * num_nodes)
num_val = int(val_proportion * num_nodes)

# Assign indices to training and validation sets
train_indices = indices[:num_train]
val_indices = indices[num_train:num_train + num_val]

# Create Boolean masks
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_indices] = True
val_mask[val_indices] = True

# Add the masks to your data object
data_summarized .train_mask = train_mask
data_summarized.val_mask = val_mask

focal_loss = FocalLoss(alpha=2, gamma=3)

# Assuming 'partition' contains the community assignments
num_communities = len(partition)  # Number of unique communities
# Initialize masks for the summarized graph
train_mask_summarized = torch.zeros(num_communities, dtype=torch.bool)
val_mask_summarized = torch.zeros(num_communities, dtype=torch.bool)

# Example: Set a portion for training and the rest for validation
num_train = int(num_communities * 0.8)  # Adjust the ratio as needed
train_mask_summarized[:num_train] = True
val_mask_summarized[num_train:] = True


# Use these new masks in your training loop
for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    out = model(data_summarized)

    # Apply focal_loss using the new masks
    loss = focal_loss(out[train_mask_summarized], data_summarized.y[train_mask_summarized])
    l1_loss = l1_lambda * l1_penalty(model)
    total_loss = loss + l1_loss  # Combining focal loss with L1 loss
    total_loss .backward()
    optimizer.step()
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            # Predict on validation set
            val_out = model(data_summarized)
            val_pred = val_out[val_mask].max(1)[1]
            val_labels = data_summarized.y[val_mask]

            # Calculate metrics
            accuracy = accuracy_score(val_labels.cpu(), val_pred.cpu(),)
            precision = precision_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)
            recall = recall_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=0)
            f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=0)

        print(f'Epoch {epoch}, Loss: {loss.item()}, Acc: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')

print("Shape of community_features_tensor:", community_features_tensor.shape)
print("Shape of community_labels_tensor:", community_labels_tensor.shape)

from collections import Counter
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx
import igraph as ig
import leidenalg
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define sensitivity parameters
k_values = [2, 4, 6, 8]
modularity_resolutions = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]
edge_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
results = []

# Scale features
scaler = StandardScaler()
selected_features = [
    'finished_timestamp',
    'collected_timestamp', 'delegated_timestamp', 'operation_metadata.operation_start',
    'agent_reported_time', 'agent_metadata.created', 'pid',
    'agent_metadata.ppid', 'agent_metadata.pid', "attack_metadata.tactic_encoded", "attack_metadata.technique_name_encoded"
]
node_features = torch.tensor(scaler.fit_transform(df[selected_features]), dtype=torch.float)

# Encode labels
labels = LabelEncoder().fit_transform(df['operation_metadata.operation_adversary'])
labels = torch.tensor(labels, dtype=torch.long)

# Define Focal Loss
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=2, gamma=3, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        return F_loss.mean() if self.reduction == 'mean' else F_loss.sum()

focal_loss = FocalLoss(alpha=2, gamma=3)
l1_lambda = 0.005  # Define L1 penalty

# Sensitivity Analysis Loop
for k in k_values:
    for resolution in modularity_resolutions:
        for threshold in edge_thresholds:
            # Step 1: Generate community structure
            G_behavior = nx.from_pandas_edgelist(df, 'agent_metadata.ppid', 'node_id')
            G_igraph = ig.Graph.TupleList(G_behavior.edges(), directed=False)
            partition = leidenalg.find_partition(G_igraph, leidenalg.ModularityVertexPartition)
            communities = [list(community) for community in partition]

            # Step 2: Extract community features and labels with consistent lengths
            community_features = []
            community_labels = []

            for idx, community in enumerate(communities):
                community_node_features = [node_features[node].numpy() for node in community if node < len(node_features)]

                if community_node_features:
                    features_array = np.stack(community_node_features)
                    community_features.append(np.mean(features_array, axis=0))

                    valid_nodes = [node for node in community if node < len(labels)]
                    if valid_nodes:
                        community_label_counts = Counter(labels[node].item() for node in valid_nodes)
                        most_common_label = community_label_counts.most_common(1)[0][0]
                        community_labels.append(most_common_label)
                    else:
                        community_labels.append(-1)  # Placeholder if no valid nodes

            # Ensure that feature and label lists are consistent in length
            if len(community_features) == len(community_labels):
                community_features_tensor = torch.tensor(
                    scaler.fit_transform(np.array(community_features)), dtype=torch.float
                )
                community_labels_tensor = torch.tensor(community_labels, dtype=torch.long)
            else:
                raise ValueError("Mismatch in number of community features and labels.")

            # Step 3: Build the GAT model
            class GAT(torch.nn.Module):
                def __init__(self, num_node_features, num_classes):
                    super(GAT, self).__init__()
                    self.conv1 = GATConv(num_node_features, 64, heads=8, concat=True, dropout=0.6)
                    self.conv2 = GATConv(64 * 8, 32, heads=8, concat=True, dropout=0.6)
                    self.conv3 = GATConv(32 * 8, 16, heads=8, concat=True, dropout=0.6)
                    self.conv4 = GATConv(16 * 8, num_classes, heads=1, concat=False, dropout=0.6)
                    self.skip_connection = torch.nn.Linear(num_node_features, num_classes)

                def forward(self, data):
                    x, edge_index = data.x, data.edge_index
                    x1 = F.relu(self.conv1(x, edge_index))
                    x2 = F.relu(self.conv2(x1, edge_index))
                    x3 = F.relu(self.conv3(x2, edge_index))
                    x4 = self.conv4(x3, edge_index)
                    skip = self.skip_connection(x)
                    return F.log_softmax(x4 + skip, dim=1)

            # Edge list construction with bounds checking
            edge_list = [
                (u, v) for community in communities for u in community for v in community
                if u != v and u < community_features_tensor.size(0) and v < community_features_tensor.size(0)
            ]
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

            # Update Data object with correct edge_index
            data_summarized = Data(x=community_features_tensor, y=community_labels_tensor, edge_index=edge_index)

            # Step 4: Define masks for training and validation
            num_communities = data_summarized.y.size(0)
            train_mask = torch.zeros(num_communities, dtype=torch.bool)
            val_mask = torch.zeros(num_communities, dtype=torch.bool)
            train_mask[:int(num_communities * 0.7)] = True
            val_mask[int(num_communities * 0.7):] = True

            # Attach the corrected masks to `data_summarized`
            data_summarized.train_mask = train_mask
            data_summarized.val_mask = val_mask

            # Training loop
            best_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
            model = GAT(num_node_features=community_features_tensor.shape[1], num_classes=len(np.unique(labels)))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

            for epoch in range(500):
                model.train()
                optimizer.zero_grad()

                out = model(data_summarized)
                loss = focal_loss(out[data_summarized.train_mask], data_summarized.y[data_summarized.train_mask])
                total_loss = loss + l1_lambda * sum(param.abs().sum() for param in model.parameters())
                total_loss.backward()
                optimizer.step()

                if epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_out = model(data_summarized)
                        val_pred = val_out[data_summarized.val_mask].max(1)[1]
                        val_labels = data_summarized.y[data_summarized.val_mask]

                        # Set zero_division=1 to avoid warnings and handle undefined metrics
                        accuracy = accuracy_score(val_labels.cpu(), val_pred.cpu())
                        precision = precision_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)
                        recall = recall_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)
                        f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)

                        if f1 > best_metrics['f1']:
                            best_metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

            results.append({
                'k': k,
                'resolution': resolution,
                'threshold': threshold,
                'accuracy': best_metrics['accuracy'],
                'precision': best_metrics['precision'],
                'recall': best_metrics['recall'],
                'f1': best_metrics['f1']
            })

# Convert results to DataFrame for easy analysis
results_df = pd.DataFrame(results)

results_df

# Sort by highest F1 score and display top results
top_results = results_df.sort_values(by='f1', ascending=False).head(10)
print(top_results)

import matplotlib.pyplot as plt

# Plot f1 scores against threshold values for a specific k and resolution
plt.figure(figsize=(10, 6))
for k_val in results_df['k'].unique():
    subset = results_df[results_df['k'] == k_val]
    plt.plot(subset['threshold'], subset['f1'], label=f'k={k_val}')

plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score by Threshold for Different k Values')
plt.legend()
plt.show()



def extract_paths_for_class(G, class_label, labels_tensor, df):
    # Convert labels tensor to a list
    labels_list = labels_tensor.tolist()

    # Map node IDs to labels using the DataFrame
    node_labels = {row['node_id']: labels_list[idx] for idx, row in df.iterrows()}

    # Filter nodes by class
    nodes_of_class = [node for node, label in node_labels.items() if label == class_label]

    # Debugging: Check the number of nodes in the specified class
    print("Number of nodes in the specified class:", len(nodes_of_class))
    if nodes_of_class:
        print("Sample nodes:", nodes_of_class[:5])

    # Find all paths starting from each node of the class
    paths = []
    for node in nodes_of_class:
        for target in G:
            for path in nx.all_simple_paths(G, source=node, target=target, cutoff=5):
                paths.append(path)
    return paths, nodes_of_class  # Return nodes_of_class for outside inspection

# Usage
class_label_encoded = 7  # The encoded label for 'operation_metadata.operation_adversary'
all_paths = extract_paths_for_class(G_behavior, class_label_encoded, labels, df)

all_paths

all_paths, nodes_of_class = extract_paths_for_class(G_behavior, class_label_encoded, labels, df)

# Check if paths are found
print("Number of paths found:", len(all_paths))
if all_paths:
    print("Sample paths:", all_paths[:5])

def aggregate_path_context(G, path, df):
    path_context = {
        'nodes': [],
        'edges': [],
        "attack_metadata.tactic_encoded": [],
        'attack_metadata.technique_name_encoded': [],
        'collected_timestamp': [],
        'finished_timestamp': [],
        'delegated_timestamp': [],
        'operation_metadata.operation_start': [],
        'agent_reported_time': [],
        'agent_metadata.created': [],
        'pid': [],
        'agent_metadata.ppid': [],
        'agent_metadata.pid': []
    }

    for i in range(len(path)):
        node = path[i]

        # Aggregate node information from the graph
        path_context['nodes'].append(G.nodes[node])

        # For the last node in the path, there won't be a next node
        if i < len(path) - 1:
            next_node = path[i + 1]
            # Aggregate edge information
            if G.has_edge(node, next_node):
                path_context['edges'].append(G[node][next_node])

        # Aggregate contextual information from the DataFrame
        node_data = df.loc[df['node_id'] == node].iloc[0]
        path_context['attack_metadata.tactic_encoded'].append(node_data['attack_metadata.tactic_encoded'])
        path_context['attack_metadata.technique_name_encoded'].append(node_data['attack_metadata.technique_name_encoded'])
        path_context['collected_timestamp'].append(node_data['collected_timestamp'])

        # Aggregate new features
        for feature in ['finished_timestamp', 'delegated_timestamp', 'operation_metadata.operation_start',
                        'agent_reported_time', 'agent_metadata.created', 'pid', 'agent_metadata.ppid',
                        'agent_metadata.pid']:
            if feature in node_data:
                path_context[feature].append(node_data[feature])
            else:
                path_context[feature].append(None)  # or some other default value

    return path_context

# Assuming all_paths contains your paths and G_behavior is your graph
aggregated_path_contexts = [aggregate_path_context(G_behavior, path, df) for path in all_paths]

# Now, aggregated_path_contexts contains the contextual information for each path

# If you want to look at the context for a specific path
specific_path = all_paths[0]  # For example, the first path
path_context = aggregate_path_context(G_behavior, specific_path, df)

# path_context now contains the aggregated data for this specific path

path_context

# Assuming aggregated_path_contexts contains your aggregated data for each path
for i, path_context in enumerate(aggregated_path_contexts):
    print(f"Path {i+1}:")

    # Define the features you want to print
    features = [
        'finished_timestamp',
        'operation_metadata.operation_adversary_encoded',
        'collected_timestamp',
        'delegated_timestamp',
        'operation_metadata.operation_start',
        'agent_reported_time',
        'agent_metadata.created',
        'pid',
        'agent_metadata.ppid',
        'agent_metadata.pid',
        'attack_metadata.tactic_encoded',
        'attack_metadata.technique_name_encoded'
    ]

    # Print each feature if available
    for feature in features:
        if feature in path_context:
            print(f"  {feature.replace('_', ' ').title()}:")
            for item in path_context[feature]:
                print(f"    {item}")
        else:
            print(f"  {feature.replace('_', ' ').title()}: Not available")

    print("\n")  # Newline for separation between paths

# Mapping of encoded numbers to tactic names
tactic_mapping = {
    0: 'collection',
    1: 'command-and-control',
    2: 'credential-access',
    3: 'defense-evasion',
    4: 'discovery',
    5: 'execution',
    6: 'exfiltration',
    7: 'impact',
    8: 'initial-access',
    9: 'lateral-movement',
    10: 'privilege-escalation'
    # Add any additional mappings if there are more encoded values
}

technique_mapping = {
    0: "Abuse Elevation Control Mechanism: Bypass User Access Control (T1548.002)",
    1: "Account Discovery: Domain Account (T1087.002)",
    2: "Account Discovery: Local Account (T1087.001)",
    3: "Application Layer Protocol: Web Protocols (T1071.001)",
    4: "Application Window Discovery (T1010)",
    5: "Archive Collected Data: Archive via Utility (T1560.001)",
    6: "Clipboard Data (T1115)",
    7: "Command and Scripting Interpreter: PowerShell (T1059.001)",
    8: "Command and Scripting Interpreter: Unix Shell (T1059.004)",
    9: "Data Manipulation: Stored Data Manipulation (T1565.001)",
    10: "Data Staged: Local Data Staging (T1074.001)",
    11: "Data Transfer Size Limits (T1030)",
    12: "Data from Local System (T1005)",
    13: "Defacement (T1491)",
    14: "Domain Trust Discovery (T1482)",
    15: "Endpoint Denial of Service (T1499)",
    16: "Exfiltration Over C2 Channel (T1041)",
    17: "File and Directory Discovery (T1083)",
    18: "Hide Artifacts: NTFS File Attributes (T1564.004)",
    19: "Hijack Execution Flow: Services File Permissions Weakness (T1574.010)",
    20: "Indicator Removal on Host: Clear Command History (T1070.003)",
    21: "Indicator Removal on Host: Timestomp (T1070.006)",
    22: "Ingress Tool Transfer (T1105)",
    23: "Lateral Tool Transfer (T1570)",
    24: "Network Service Scanning (T1046)",
    25: "Network Share Discovery (T1135)",
    26: "Network Sniffing (T1040)",
    27: "OS Credential Dumping (T1003)",
    28: "OS Credential Dumping: /etc/passwd and /etc/shadow (T1003.008)",
    29: "OS Credential Dumping: LSASS Memory (T1003.001)",
    30: "Password Policy Discovery (T1201)",
    31: "Password Policy Discovery for a domain (T1201)",
    32: "Peripheral Device Discovery (T1120)",
    33: "Permission Groups Discovery: Domain Groups (T1069.002)",
    34: "Permission Groups Discovery: Local Groups (T1069.001)",
    35: "Phishing: Spearphishing Attachment (T1566.001)",
    36: "Process Discovery (T1057)",
    37: "Process Injection: Dynamic-link Library Injection (T1055.001)",
    38: "Process Injection: Portable Executable Injection (T1055.002)",
    39: "Query Registry (T1012)",
    40: "Remote System Discovery (T1018)",
    41: "Resource Hijacking (T1496)",
    42: "Scheduled Transfer (T1029)",
    43: "Screen Capture (T1113)",
    44: "Software Discovery (T1518)",
    45: "Software Discovery: Security Software Discovery (T1518.001)",
    46: "System Information Discovery (T1082)",
    47: "System Network Configuration Discovery (T1016)",
    48: "System Network Connections Discovery (T1049)",
    49: "System Owner/User Discovery (T1033)",
    50: "System Service Discovery (T1007)",
    51: "System Services: Service Execution (T1569.002)",
    52: "System Time Discovery (T1124)",
    53: "Unsecured Credentials: Bash History (T1552.003)",
    54: "Unsecured Credentials: Credentials in Registry (T1552.002)",
    55: "Unsecured Credentials: Private Keys (T1552.004)",
    56: "Virtualization/Sandbox Evasion: System Checks (T1497.001)",
    57: "WMIC (T1047)",
    58: "host discovery (TA0007)"
}



# Assuming aggregated_path_contexts contains your aggregated data for each path
for i, path_context in enumerate(aggregated_path_contexts):
    print(f"Path {i+1}:")

    # Define the features you want to print
    features = [
        # ... other features ...
        'finished_timestamp',
        'collected_timestamp',
        'delegated_timestamp',
        'operation_metadata.operation_start',
        'agent_reported_time',
        'agent_metadata.created',
        'pid',
        'agent_metadata.ppid',
        'agent_metadata.pid',
        'attack_metadata.tactic_encoded',
        'attack_metadata.technique_name_encoded'
    ]


     # Print each feature if available
    for feature in features:
        if feature in path_context:
            print(f"  {feature.replace('_', ' ').title()}:")

            for item in path_context[feature]:
                if feature == 'attack_metadata.tactic_encoded':
                    # Decode the tactic using the mapping
                    decoded_value = tactic_mapping.get(item, 'Unknown Tactic')
                    print(f"    {decoded_value}")
                elif feature == 'attack_metadata.technique_name_encoded':
                    # Decode the technique using the mapping
                    decoded_value = technique_mapping.get(item, 'Unknown Technique')
                    print(f"    {decoded_value}")
                else:
                    # For other features, just print the item
                    print(f"    {item}")
        else:
            print(f"  {feature.replace('_', ' ').title()}: Not available")

    print("\n")  # Newline for separation between paths
    # for feature in features:
    #     if feature in path_context:
    #         print(f"  {feature.replace('_', ' ').title()}:")

    #         for item in path_context[feature]:
    #             if feature == 'attack_metadata.tactic_encoded':
    #                 # Decode the tactic using the mapping
    #                 decoded_value = tactic_mapping.get(item, 'Unknown Tactic')
    #                 print(f"    {decoded_value}")
    #             else:
    #                 # For other features, just print the item
    #                 print(f"    {item}")
    #     else:
    #         print(f"  {feature.replace('_', ' ').title()}: Not available")

    # print("\n")  # Newline for separation between paths

import matplotlib.pyplot as plt
import networkx as nx

subgraph_nodes = set(itertools.chain.from_iterable(all_paths))
G_sub = G_behavior.subgraph(subgraph_nodes)

# Position nodes using one of the layout algorithms e.g., spring_layout
pos = nx.spring_layout(G_sub, seed=42)  # Seed for reproducible layout

# Draw the nodes
nx.draw_networkx_nodes(G_sub, pos, node_color='lightblue', node_size=50)

# Draw the edges
nx.draw_networkx_edges(G_sub, pos, edgelist=G_sub.edges(), edge_color='gray')

# Optionally, draw edge labels or node labels
# nx.draw_networkx_edge_labels(G_sub, pos, edge_labels={(u, v): d['weight'] for u, v, d in G_sub.edges(data=True)})
# nx.draw_networkx_labels(G_sub, pos)

plt.title("Graph Visualization of Paths")
plt.show()

import matplotlib.pyplot as plt

num_classes = len(torch.unique(labels).tolist())

# Generate a color palette that can accommodate the number of unique classes
colors = plt.cm.get_cmap('tab20', num_classes)

# Create a color map: class -> color
class_to_color = {i: colors(i / num_classes) for i in range(num_classes)}

def gat_predict(input_data):
    # Convert input data to PyTorch tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float)

    # Create a dummy edge index (since it's not used in prediction but required by the model)
    dummy_edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Create a data object
    dummy_data = Data(x=input_tensor, edge_index=dummy_edge_index)

    # Make sure the model is in evaluation mode
    model.eval()

    # Get predictions
    with torch.no_grad():
        logits = model(dummy_data)
        probabilities = F.softmax(logits, dim=1).numpy()

    return probabilities

pip install lime

import lime

from lime import lime_tabular

# Assuming community_features_tensor is your feature set
explainer = lime_tabular.LimeTabularExplainer(
    training_data=community_features_tensor.numpy(),
    feature_names=selected_features,
    class_names=['Class1', 'Class2'],  # Update with your actual class names
    mode='classification'
)

# Explain the first instance
exp = explainer.explain_instance(
    data_row=community_features_tensor[0].numpy(),
    predict_fn=gat_predict
)

# Show the explanation
exp.show_in_notebook(show_table=True, show_all=False)

def gat_predict(input_data):
    # Convert input data to PyTorch tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float)

    # Create a dummy edge index
    dummy_edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Create a data object for the GAT model
    dummy_data = Data(x=input_tensor, edge_index=dummy_edge_index)

    # Make sure the model is in evaluation mode
    model.eval()

    # Get predictions
    with torch.no_grad():
        logits = model(dummy_data)
        probabilities = F.softmax(logits, dim=1).numpy()

    return probabilities

from lime import lime_tabular

# Prepare the explainer
explainer = lime_tabular.LimeTabularExplainer(
    training_data=community_features_tensor.numpy(),
    feature_names=selected_features,
    class_names=[str(i) for i in range(num_classes)],  
    mode='classification'
)

# Choose a specific node/community for explanation
node_index_to_explain = 0  # for example, explaining the first node/community

# Explain the instance
exp = explainer.explain_instance(
    data_row=community_features_tensor[node_index_to_explain].numpy(),
    predict_fn=gat_predict
)

# Visualize the explanation
exp.show_in_notebook(show_table=True, show_all=False)

num_supernodes = len(G_summary.nodes())
num_superedges = len(G_summary.edges())

print(f"Number of supernodes in the summarized graph: {num_supernodes}")
print(f"Number of superedges in the summarized graph: {num_superedges}")

def estimate_summary_graph_memory_usage(num_supernodes, avg_supernode_id_length, avg_list_elements, avg_element_length, num_superedges,
                                        avg_memory_per_char=74, avg_memory_per_edge=64):
    """
    Estimate the memory usage of the summary graph.

    :param num_supernodes: Number of supernodes in the graph.
    :param avg_supernode_id_length: Average length of the supernode identifiers.
    :param avg_list_elements: Average number of elements in each attribute list for a supernode.
    :param avg_element_length: Average length of strings in the attribute lists.
    :param num_superedges: Number of superedges in the summary graph.
    :param avg_memory_per_char: Average memory usage per character in a node identifier (default is 74 bytes).
    :param avg_memory_per_edge: Average memory usage per edge (default is 64 bytes).
    :return: Estimated memory usage of the summary graph in bytes.
    """
    # Calculate total memory for supernode identifiers
    total_memory_for_supernode_ids = num_supernodes * avg_supernode_id_length * avg_memory_per_char

    # Estimate memory for attribute lists in supernodes
    # Assuming multiple attributes, each being a list of strings
    num_attributes = 10  # number of attributes per supernode (based on the provided code)
    memory_per_attribute = avg_list_elements * avg_element_length * avg_memory_per_char
    total_memory_for_attributes = num_supernodes * num_attributes * memory_per_attribute

    # Calculate total memory for edges
    total_memory_for_edges = num_superedges * avg_memory_per_edge

    # Total estimated memory usage
    return total_memory_for_supernode_ids + total_memory_for_attributes + total_memory_for_edges




