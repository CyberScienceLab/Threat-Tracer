
df = pd.read_csv('Clean_dataset.csv')



# Example of label encoding for categorical features
label_encoders = {}
for column in categorical_features:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Start preprocessing

# Define Nodes:
df['node_id'] = range(len(df))  # Unique identifier for each command instance

# Define Edges:
edges = []
for i in range(len(df) - 1):
    edge = (df['node_id'][i], df['node_id'][i + 1])
    edges.append(edge)
edges = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Extract Node Features:
selected_features = [
    'finished_timestamp', 'operation_metadata.operation_adversary_encoded',
    'collected_timestamp', 'delegated_timestamp', 'operation_metadata.operation_start',
    'agent_reported_time', 'agent_metadata.created', 'pid',
    'agent_metadata.ppid', 'agent_metadata.pid', "attack_metadata.tactic_encoded", "attack_metadata.technique_name_encoded"
]
node_features = df[selected_features]
node_features = torch.tensor(node_features.values, dtype=torch.float)

# Encode Labels:
labels = LabelEncoder().fit_transform(df['operation_metadata.operation_adversary'])
labels = torch.tensor(labels, dtype=torch.long)

# Create the Data Object for PyTorch Geometric:
from torch_geometric.data import Data
data = Data(x=node_features, edge_index=edges, y=labels)

edge_index = torch.tensor(edges, dtype=torch.long)

print(type(node_features))

# If it's a DataFrame, the following should work
if isinstance(node_features, pd.DataFrame):
    x = torch.tensor(node_features.values, dtype=torch.float)
else:
    # Handle the case where node_features is not a DataFrame
    print("node_features is not a DataFrame")

# Check if 'labels' is already a tensor
if not isinstance(labels, torch.Tensor):
    labels = torch.tensor(labels, dtype=torch.long)
else:
    # If it's already a tensor, use it directly
    labels = labels.clone().detach()

# # Convert node features to PyTorch tensor
# x = torch.tensor(node_features.values, dtype=torch.float)

# Convert labels to PyTorch tensor
# y = torch.tensor(labels, dtype=torch.long)

# Convert edges to PyTorch tensor and format it correctly
# edge_index = torch.tensor(edges, dtype=torch.long)

# Create a PyTorch Geometric data object
data = Data(x=node_features, edge_index=edge_index.t().contiguous(), y=labels)

print(data.edge_index.shape)

print(data.num_nodes)
print(data.x.shape[0])  # Number of node features

data.edge_index = data.edge_index.t()

# # # Define a simple GCN model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, len(df['operation_metadata.operation_adversary'].unique()))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)


import torch
from torch_geometric.nn import GCNConv, BatchNorm, GATConv
import torch.nn.functional as F


from sklearn.utils.class_weight import compute_class_weight

# Assuming 'labels' is your tensor of labels
class_weights = compute_class_weight('balanced', classes=np.unique(labels.numpy()), y=labels.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Instantiate the model
model = GCN()

# Define loss function and optimizer
# criterion = torch.nn.CrossEntropyLoss()

# Apply to your loss function
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

import numpy as np

# Total number of nodes
num_nodes = data.num_nodes

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
data.train_mask = train_mask
data.val_mask = val_mask

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
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

# Assuming FocalLoss class is already defined

# Initialize Focal Loss
focal_loss = FocalLoss(alpha=4, gamma=5)

for epoch in range(400):
    model.train()
    optimizer.zero_grad()
    out = model(data)

    # Replace criterion with focal_loss
    loss = focal_loss(out[train_mask], data.y[train_mask]) # Ensure the target is properly formatted
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            # Predict on validation set
            val_out = model(data)
            val_pred = val_out[val_mask].max(1)[1]
            val_labels = data.y[val_mask]

            # Calculate metrics
            accuracy = accuracy_score(val_labels.cpu(), val_pred.cpu())
            precision = precision_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)
            recall = recall_score(val_labels.cpu(), val_pred.cpu(), average='weighted')
            f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average='weighted')

        print(f'Epoch {epoch}, Loss: {loss.item()}, Acc: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')

from sklearn.preprocessing import StandardScaler
# Assuming 'data.x' is your node feature matrix
scaler = StandardScaler()

# Fit and transform the features
data.x = scaler.fit_transform(data.x.numpy())

# Convert it back to a tensor
data.x = torch.tensor(data.x, dtype=torch.float)

import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class ImprovedGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(ImprovedGAT, self).__init__()
        self.conv1 = GATConv(num_node_features, 128, heads=16, concat=True, dropout=0.5)
        self.conv2 = GATConv(128* 16, 64, heads=16, concat=True, dropout=0.5)
        self.conv3 = GATConv(64 * 16, 32, heads=16, concat=True, dropout=0.5)
        self.conv4 = GATConv(32 * 16, num_classes, heads=1, concat=False, dropout=0.5)
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

# Instantiate the model
num_classes = len(df['operation_metadata.operation_adversary'].unique())
model = ImprovedGAT(num_node_features=data.num_node_features, num_classes=num_classes)

# Define loss function and optimizer
# criterion = torch.nn.CrossEntropyLoss()

# Apply to your loss function
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class ImprovedGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(ImprovedGAT, self).__init__()
        self.conv1 = GATConv(num_node_features, 128, heads=16, concat=True, dropout=0.5)
        self.conv2 = GATConv(128* 16, 64, heads=16, concat=True, dropout=0.5)
        self.conv3 = GATConv(64 * 16, 32, heads=16, concat=True, dropout=0.5)
        self.conv4 = GATConv(32 * 16, num_classes, heads=1, concat=False, dropout=0.5)
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

# Load the entire model
model = torch.load('improved_gat_model.pth')

# Set it to evaluation mode
model.eval()

# Ablation study


import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.nn import GATConv

# Improved GAT model definition
class DynamicGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(DynamicGAT, self).__init__()
        self.conv1 = GATConv(num_node_features, 128, heads=16, concat=True, dropout=0.5)
        self.conv2 = GATConv(128 * 16, 64, heads=16, concat=True, dropout=0.5)
        self.conv3 = GATConv(64 * 16, 32, heads=16, concat=True, dropout=0.5)
        self.conv4 = GATConv(32 * 16, num_classes, heads=1, concat=False, dropout=0.5)
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

# Function to train and evaluate the model
def train_and_evaluate(data, train_mask, val_mask, num_features, num_classes):
    # Initialize a new model with the correct input size
    model = DynamicGAT(num_node_features=num_features, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize Focal Loss
    focal_loss = FocalLoss(alpha=4, gamma=5)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data)

        # Replace criterion with focal_loss
        loss = focal_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Predict on validation set
                val_out = model(data)
                val_pred = val_out[val_mask].max(1)[1]
                val_labels = data.y[val_mask]

                # Calculate metrics
                accuracy = accuracy_score(val_labels.cpu(), val_pred.cpu())
                precision = precision_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)
                recall = recall_score(val_labels.cpu(), val_pred.cpu(), average='weighted')
                f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average='weighted')

            print(f'Epoch {epoch}, Loss: {loss.item()}, Acc: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')

    return accuracy, precision, recall, f1

# Function to perform ablation study
def ablation_study(data, train_mask, val_mask, num_classes):
    num_features = data.x.size(1)
    results = []

    # Train with all features
    print("Training with all features...")
    all_feature_metrics = train_and_evaluate(data, train_mask, val_mask, num_features, num_classes)
    results.append(("All Features", *all_feature_metrics))

    # Systematically exclude one feature at a time
    for feature_idx in range(num_features):
        print(f"Excluding feature {feature_idx}...")

        # Exclude the current feature
        modified_x = torch.cat([data.x[:, :feature_idx], data.x[:, feature_idx+1:]], dim=1)
        modified_data = Data(x=modified_x, edge_index=data.edge_index, y=data.y)

        # Train and evaluate
        metrics = train_and_evaluate(modified_data, train_mask, val_mask, num_features - 1, num_classes)
        results.append((f"Feature {feature_idx} Excluded", *metrics))

    return results

# Assuming you already have the dataset `data`, `train_mask`, and `val_mask`
# Also assuming `data.num_node_features` gives the number of features

# Define the number of classes
num_classes = len(df['operation_metadata.operation_adversary'].unique())

# Run the ablation study
ablation_results = ablation_study(data, train_mask, val_mask, num_classes)

# Convert the results into a DataFrame for better visualization
import pandas as pd
results_df = pd.DataFrame(ablation_results, columns=["Feature Set", "Accuracy", "Precision", "Recall", "F1 Score"])

# Display results
print(results_df)

# Assuming you have the ablation study results in a DataFrame 'results_df'
results_df['F1 Score Drop'] = results_df['F1 Score'].iloc[0] - results_df['F1 Score']
top_features_df = results_df.sort_values(by='F1 Score Drop', ascending=False).head(3)
top_features = [int(feature.split(' ')[1]) for feature in top_features_df['Feature Set']]
print("Top 3 Most Important Features:", top_features)

from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Assuming node_features and labels are available
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(node_features.numpy(), labels.numpy())

# Extract feature importances
importances = rf_model.feature_importances_

# Rank the features
feature_ranking = np.argsort(importances)[::-1]
top_features = feature_ranking[:3]
print("Top 3 Most Important Features:", top_features)

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Function to train and evaluate the model
def train_and_evaluate(data, train_mask, val_mask, num_features, num_classes):
    # Initialize a new model with the correct input size
    model = DynamicGAT(num_node_features=num_features, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize Focal Loss
    focal_loss = FocalLoss(alpha=4, gamma=5)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data)

        # Replace criterion with focal_loss
        loss = focal_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Predict on validation set
                val_out = model(data)
                val_pred = val_out[val_mask].max(1)[1]
                val_labels = data.y[val_mask]

                # Calculate metrics
                accuracy = accuracy_score(val_labels.cpu(), val_pred.cpu())
                precision = precision_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)
                recall = recall_score(val_labels.cpu(), val_pred.cpu(), average='weighted')
                f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average='weighted')

            print(f'Epoch {epoch}, Loss: {loss.item()}, Acc: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')

    return accuracy, precision, recall, f1

# Function to perform incremental feature addition
def incremental_feature_addition(data, train_mask, val_mask, num_classes, top_features):
    results = []

    # Add features one by one incrementally
    for i in range(1, len(top_features) + 1):
        print(f"Using top {i} features: {top_features[:i]}...")

        # Select the subset of features based on the order of importance
        selected_features = top_features[:i]
        modified_x = data.x[:, selected_features]

        # Create new data object with selected features
        modified_data = Data(x=modified_x, edge_index=data.edge_index, y=data.y)

        # Train and evaluate
        metrics = train_and_evaluate(modified_data, train_mask, val_mask, len(selected_features), num_classes)
        results.append((f"Top {i} Features", *metrics))

    return results

# Assuming you have identified the top features based on your ablation study
# The top 3 most important features based on ablation study: Feature 1, Feature 8, Feature 10
top_features = [1, 8, 10]  # Feature indices that were identified as most important

# Run the incremental feature addition
incremental_results = incremental_feature_addition(data, train_mask, val_mask, num_classes, top_features)

# Convert the results into a DataFrame for better visualization
results_df = pd.DataFrame(incremental_results, columns=["Feature Set", "Accuracy", "Precision", "Recall", "F1 Score"])

# Display results
print(results_df)


# Function to perform multiple runs and store the results
def multiple_runs(data, train_mask, val_mask, num_classes, num_runs=2):
    all_metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}
    for run in range(num_runs):
        metrics = train_and_evaluate(data, train_mask, val_mask, num_classes, random_seed=run)
        all_metrics['Accuracy'].append(metrics[0])
        all_metrics['Precision'].append(metrics[1])
        all_metrics['Recall'].append(metrics[2])
        all_metrics['F1 Score'].append(metrics[3])
    return all_metrics

# Function to perform ablation study with multiple runs and T-test
def ablation_study_t_test(data, train_mask, val_mask, num_classes, num_runs=10):
    num_features = data.x.size(1)
    all_feature_metrics = multiple_runs(data, train_mask, val_mask, num_classes, num_runs)

    t_test_results = {'Feature': [], 'Metric': [], 'T-Statistic': [], 'P-Value': []}

    # Loop through each feature to exclude and calculate T-test
    for feature_idx in range(num_features):
        print(f"Excluding feature {feature_idx}...")

        modified_x = torch.cat([data.x[:, :feature_idx], data.x[:, feature_idx+1:]], dim=1)
        modified_data = Data(x=modified_x, edge_index=data.edge_index, y=data.y)

        excluded_feature_metrics = multiple_runs(modified_data, train_mask, val_mask, num_classes, num_runs)

        # Perform T-test for each metric
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
            t_stat, p_value = ttest_ind(all_feature_metrics[metric], excluded_feature_metrics[metric], equal_var=False)
            t_test_results['Feature'].append(f"Feature {feature_idx} Excluded")
            t_test_results['Metric'].append(metric)
            t_test_results['T-Statistic'].append(t_stat)
            t_test_results['P-Value'].append(p_value)

    # Convert results to DataFrame for better visualization
    t_test_results_df = pd.DataFrame(t_test_results)
    return t_test_results_df

# Assuming you already have the dataset `data`, `train_mask`, and `val_mask`
num_classes = len(df['operation_metadata.operation_adversary'].unique())
t_test_results_df = ablation_study_t_test(data, train_mask, val_mask, num_classes)
print(t_test_results_df)


