import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import networkx as nx
from sklearn.cluster import SpectralClustering

pip install torch-geometric

df = pd.read_csv('CKC_dataset.csv')

df

# Encode the categorical data
categorical_columns = ['ckc_step']
for column in categorical_columns:
    df[column + '_encoded'] = pd.Categorical(df[column]).codes

df

# Extract Node Features:
selected_features = [
    'finished_timestamp', 'operation_metadata.operation_adversary_encoded',
    'collected_timestamp', 'delegated_timestamp', 'operation_metadata.operation_start',
    'agent_reported_time', 'agent_metadata.created', 'pid',
    'agent_metadata.ppid', 'agent_metadata.pid', 'ckc_step_encoded'  # Replace tactic and technique with 'ckc_step_num'
]
node_features = df[selected_features]

node_features

# Define Nodes:
df = df[selected_features].copy()  # Create a copy of the DataFrame
df['node_id'] = range(len(df))  # Unique identifier for each command instance

# Define Edges:
edges = []
for i in range(len(df) - 1):
    edge = (df['node_id'][i], df['node_id'][i + 1])
    edges.append(edge)
edges = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Encode Labels:
labels = LabelEncoder().fit_transform(df['operation_metadata.operation_adversary_encoded'])
labels = torch.tensor(labels, dtype=torch.long)

print (labels)

G_behavior = nx.DiGraph()

# Add nodes
for node_id in df['node_id']:
    G_behavior.add_node(node_id)

# Add edges
for edge in edges.t().numpy():
    G_behavior.add_edge(edge[0], edge[1])

# Add node attributes from DataFrame
for idx, row in df.iterrows():
    node_id = row['node_id']
    node_attrs = row.to_dict()
    G_behavior.nodes[node_id].update(node_attrs)

G_behavior

import matplotlib.pyplot as plt

pos = nx.spring_layout(G_behavior)

# Detailed plot
plt.figure(figsize=(15, 10))  # Adjust the size as needed
nx.draw_networkx_nodes(G_behavior, pos, node_color='lightblue')
nx.draw_networkx_edges(G_behavior, pos, edge_color='gray')
nx.draw_networkx_labels(G_behavior, pos)

# If you want to add edge labels
edge_labels = nx.get_edge_attributes(G_behavior, 'label')
nx.draw_networkx_edge_labels(G_behavior, pos, edge_labels=edge_labels)

plt.title("Detailed Behavior Graph")
plt.show()

# List of node IDs you're interested in
nodes_of_interest = [16635, 19531, 16679, 18776, 17029, 14508, 12630, 23164, 12775, 18522]

# Iterate over the nodes and print their attributes
for node_id in nodes_of_interest:
    if node_id in G_behavior.nodes:
        print(f"Attributes of Node {node_id}:")
        for attr, value in G_behavior.nodes[node_id].items():
            print(f"  {attr}: {value}")
        print("\n")
    else:
        print(f"Node {node_id} not found in the graph.\n")

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
    'finished_timestamps': set(),
    'operation_metadata.operation_adversary_encoded':set(),
    'collected_timestamp': set(),
    'delegated_timestamp': set(),
    'operation_metadata.operation_start': set(),
    'agent_reported_time': set(),
    'agent_metadata.created': set(),
    'pid': set(),
    'agent_metadata.ppid': set(),
    'agent_metadata.pid': set(),
    'ckc_step_encoded': set()
} for i in set(community_map.values())}

# Plural forms of node types
plural_map = {
    'operation': 'operations',
    'log': 'logs',
    'pid':'pids',
    'platform': 'platforms',
    'executor': 'executors',
    'ability': 'abilities',
    'ckc_step_encoded':'ckc_step_encoded'
}

# Assign nodes to communities and aggregate features
for node, community in community_map.items():
    community_info[community]['nodes'].add(node)
    node_data = G_behavior.nodes[node]

    # Aggregate features
    community_info[community]['finished_timestamps'].add(node_data.get('finished_timestamp'))
    community_info[community]['collected_timestamp'].add(node_data.get('collected_timestamp'))
    community_info[community]['delegated_timestamp'].add(node_data.get('delegated_timestamp'))
    community_info[community]['operation_metadata.operation_start'].add(node_data.get('operation_metadata.operation_start'))
    community_info[community]['agent_reported_time'].add(node_data.get('agent_reported_time'))
    community_info[community]['agent_metadata.created'].add(node_data.get('agent_metadata.created'))
    community_info[community]['pid'].add(node_data.get('pid'))
    community_info[community]['agent_metadata.ppid'].add(node_data.get('agent_metadata.ppid'))
    community_info[community]['agent_metadata.pid'].add(node_data.get('agent_metadata.pid'))
    community_info[community]["ckc_step_encoded"].add(node_data.get("ckc_step_encoded"))
    community_info[community]['operation_metadata.operation_adversary_encoded'].add(node_data.get('operation_metadata.operation_adversary_encoded'))


    # Check if the node is a log and aggregate command and status
    if node_data.get('type') == 'log':
        community_info[community]['commands'].add(node_data.get('command', ''))
        community_info[community]['statuses'].add(node_data.get('status', ''))
    else:
        feature_type = node_data.get('type')
        if feature_type:
            plural_feature_type = plural_map.get(feature_type, feature_type + 's')
            community_info[community][plural_feature_type].add(node)


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

# Print the aggregated community information and the summary graph edges
for community_id, info in community_info.items():
    print(f"Community {community_id}:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print("\n")


# Stop measuring time
end_time = time.time()

# Calculate the time taken to create the graph
time_taken = end_time - start_time
print(f"Time taken to create the graph: {time_taken:.2f} seconds")

print("Summary Graph Edges:")
print(G_summary.edges())

# List of node IDs you're interested in
nodes_of_interest = [16635, 19531, 16679, 18776, 17029, 14508, 12630, 23164, 12775, 18522]

# Iterate over the nodes and print their attributes
for node_id in nodes_of_interest:
    if node_id in G_behavior.nodes:
        print(f"Attributes of Node {node_id}:")
        for attr, value in G_behavior.nodes[node_id].items():
            print(f"  {attr}: {value}")
        print("\n")
    else:
        print(f"Node {node_id} not found in the graph.\n")

# Print the aggregated community information
for community_id, info in community_info.items():
    print(f"Community {community_id}:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print("\n")

G_summary.edges()

import matplotlib.pyplot as plt

# Choose a layout for the graph
pos = nx.spring_layout(G_summary, k=0.5, iterations=20)

# Draw the graph
nx.draw(G_summary, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray', font_size=10, font_weight='bold')

# Optionally, add labels to the supernodes
for node, data in G_summary.nodes(data=True):
    label = f"{node}\nOps: {len(data.get('operations', []))}\nTechs: {len(data.get('ckc_steps', []))}"
    plt.text(pos[node][0], pos[node][1], label, fontsize=9, ha='center', va='center')


# Save the figure with a high DPI
plt.savefig('CKC_summarization.png', dpi=400, bbox_inches='tight')

# Show the plot
plt.title("Summarized Graph Visualization")
plt.show()

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

# Example assumptions (should be replaced with actual data for accurate estimation)
num_supernodes = 39  # Number of supernodes
avg_supernode_id_length = 20  # Average length of supernode identifiers
avg_list_elements = 10  # Average number of elements in attribute lists
avg_element_length = 15  # Average length of strings in attribute lists
num_superedges = 38  # Number of superedges

estimated_memory = estimate_summary_graph_memory_usage(num_supernodes, avg_supernode_id_length, avg_list_elements, avg_element_length, num_superedges)
print(f"Estimated Memory Usage: {estimated_memory / (1024 * 1024)} MB")  # Convert bytes to Megabytes for readability

# community_features = {}

# for idx, community in enumerate(partition):
#     # Extract node features for nodes in this community
#     community_node_features = [node_features.iloc[node].to_numpy() for node in community if node < len(node_features)]

#     # Ensure community has nodes and the feature dimensions are consistent
#     if community_node_features:
#         # Convert list of numpy arrays to a numpy array
#         features_array = np.stack(community_node_features)
#         # Aggregate features using mean
#         community_features[idx] = np.mean(features_array, axis=0)
community_features = {}

for idx, community in enumerate(partition):
    # Extract node features for nodes in this community
    community_node_features = [node_features.iloc[node].to_numpy() for node in community if node < len(node_features)]

    # Ensure community has nodes and the feature dimensions are consistent
    if community_node_features:
        # Convert list of numpy arrays to a numpy array
        features_array = np.stack(community_node_features)
        # Aggregate features using mean
        community_features[idx] = np.mean(features_array, axis=0)

community_features

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(community_features_array)  # Assuming community_features_array is your data array

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the features
scaled_features = scaler.fit_transform(node_features)

# Convert the scaled features to a PyTorch tensor
node_features = torch.tensor(scaled_features, dtype=torch.float)

# Assuming community_features is a dictionary or similar structure
community_features_array = np.array([community_features[i] for i in range(len(community_features))])

# Scale the community features
scaled_community_features = scaler.fit_transform(community_features_array)

# Convert the scaled features to a PyTorch tensor
community_features_tensor = torch.tensor(scaled_community_features, dtype=torch.float)

community_features_tensor

means = community_features_tensor.mean(dim=0)
stds = community_features_tensor.std(dim=0)
print("Means:", means)
print("Standard Deviations:", stds)

import matplotlib.pyplot as plt

# Assuming community_features_tensor is a PyTorch tensor
features_numpy = community_features_tensor.numpy()
plt.hist(features_numpy, bins=30)
plt.title("Feature Distributions after Standard Scaling")
plt.xlabel("Feature values")
plt.ylabel("Frequency")
plt.show()

print(list(G_summary.edges())[:5])  # Print the first 5 edges to inspect their format

# Extract unique node names from the edges
unique_nodes = set()
for edge in G_summary.edges():
    unique_nodes.update(edge)

# Create a mapping from node names to integers
node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}

# Print the mapping to verify
print(node_mapping)

# Assuming node_mapping is your mapping from supernode names to integers
print("Node mapping:", node_mapping)

# Convert edges to integer format using the mapping
integer_edges = [(node_mapping[edge[0]], node_mapping[edge[1]]) for edge in G_summary.edges()]

# Debugging print to check the converted edges
print("Converted edges (first 5):", integer_edges[:5])

edge_index = torch.tensor(integer_edges, dtype=torch.long).t().contiguous()

# Final check of edge_index tensor
print("Edge index tensor:", edge_index)

# Convert edges to integer format using the mapping
integer_edges = [(node_mapping[edge[0]], node_mapping[edge[1]]) for edge in G_summary.edges()]

# Print the first few converted edges to verify
print(integer_edges[:5])

edge_index = torch.tensor(integer_edges, dtype=torch.long).t().contiguous()

edge_index

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

data_summarized

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, 64, heads=8, concat=True, dropout=0.6)
        self.conv2 = GATConv(64* 8, 32, heads=8, concat=True, dropout=0.6)
        self.conv3 = GATConv(32 * 8, 8, heads=8, concat=True, dropout=0.6)
        self.conv4 = GATConv(8 * 8, num_classes, heads=1, concat=False, dropout=0.6)
        self.skip_connection = torch.nn.Linear(num_node_features, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.5, training=self.training)

        x3 = self.conv3(x2, edge_index)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, p=0.5, training=self.training)

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

l1_lambda = 0.001

from sklearn.utils.class_weight import compute_class_weight

# # Assuming 'labels' is your tensor of labels
class_weights = compute_class_weight('balanced', classes=np.unique(labels.numpy()), y=labels.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Instantiate the model
from torch_geometric.nn import GATConv
from torch.optim.lr_scheduler import StepLR

num_classes = len(df['operation_metadata.operation_adversary_encoded'].unique())
model = GAT(num_node_features=community_features_tensor.shape[1], num_classes=num_classes)

# Define loss function and optimizer
# criterion = torch.nn.CrossEntropyLoss()

# Apply to your loss function
# criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
class_weights = compute_class_weight('balanced', classes=np.unique(labels.numpy()), y=labels.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)  # Add L2 regularization here

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
train_proportion = 0.7
val_proportion = 0.3

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

focal_loss = FocalLoss(alpha=4, gamma=6)

# Assuming 'partition' contains the community assignments
num_communities = len(partition)  # Number of unique communities
# Initialize masks for the summarized graph
train_mask_summarized = torch.zeros(num_communities, dtype=torch.bool)
val_mask_summarized = torch.zeros(num_communities, dtype=torch.bool)

# Example: Set a portion for training and the rest for validation
num_train = int(num_communities * 0.7)  # Adjust the ratio as needed
train_mask_summarized[:num_train] = True
val_mask_summarized[num_train:] = True

early_stopping_patience = 10
best_loss = float('inf')
epochs_no_improve = 0
# Use these new masks in your training loop
for epoch in range(400):
    model.train()
    optimizer.zero_grad()
    out = model(data_summarized)

    # Apply focal_loss using the new masks
    loss = focal_loss(out[train_mask_summarized], data_summarized.y[train_mask_summarized])
    l1_loss = l1_lambda * l1_penalty(model)
    total_loss = loss + l1_loss  # Combining focal loss with L1 loss
    total_loss .backward()
    optimizer.step()
    if epoch % 1 == 0:
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

torch.save(model, 'model-2.pth')


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
        'collected_timestamp': [],
        'finished_timestamp': [],
        'delegated_timestamp': [],
        'operation_metadata.operation_start': [],
        'agent_reported_time': [],
        'agent_metadata.created': [],
        'pid': [],
        'agent_metadata.ppid': [],
        'agent_metadata.pid': [] ,
         'ckc_step_encoded':[]
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
        path_context['ckc_step_encoded'].append(node_data['ckc_step_encoded'])
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
        'collected_timestamp',
        'delegated_timestamp',
        'operation_metadata.operation_start',
        'agent_reported_time',
        'agent_metadata.created',
        'pid',
        'agent_metadata.ppid',
        'agent_metadata.pid',
        'ckc_step_encoded'
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

# CKC step mapping
ckc_step_mapping = {
    0: 'Actions on Objectives',
    1: 'Command and Control',
    2: 'Delivery',
    3: 'Exploitation',
    4: 'Installation',
    5: 'Reconnaissance'
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
        'ckc_step_encoded'  # Assuming this is the feature with the CKC step encoding
    ]

    # Print each feature if available
    for feature in features:
        if feature in path_context:
            print(f"  {feature.replace('_', ' ').title()}:")

            for item in path_context[feature]:
                if feature == 'ckc_step_encoded':
                    # Decode the CKC step using the mapping
                    decoded_value = ckc_step_mapping.get(item, 'Unknown CKC Step')
                    print(f"    {decoded_value}")
                else:
                    # For other features, just print the item
                    print(f"    {item}")
        else:
            print(f"  {feature.replace('_', ' ').title()}: Not available")

    print("\n")  # Newline for separation between paths

