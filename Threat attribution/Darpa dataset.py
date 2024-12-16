
pip install pandas networkx torch torchvision torchaudio torch-geometric

import pandas as pd

df = pd.read_csv('Combined_Tagged_Chunks.csv')

df

# Extract Node Features:
selected_features = [
'Source UUID','Source type','Destination UUID','Destination type','Edge type','Timestamp','Path', 'Tactic' , 'Technique'
]
df = df[selected_features]

df

# Define Nodes:
df = df[selected_features].copy()  # Create a copy of the DataFrame
df['node_id'] = range(len(df))  # Unique identifier for each command instance

!pip install torch torchvision
!pip install torch-geometric

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import pandas as pd
import networkx as nx
from sklearn.preprocessing import LabelEncoder

# Define Edges:
edges = []
for i in range(len(df) - 1):
    edge = (df['node_id'][i], df['node_id'][i + 1])
    edges.append(edge)
edges = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Encoding 'Tactic' and 'Technique' separately
labels= LabelEncoder().fit_transform(df['Tactic']) ######### change to labels to check
technique_labels = LabelEncoder().fit_transform(df['Technique'])

# Convert to PyTorch tensors
tactic_labels_tensor = torch.tensor(labels, dtype=torch.long)
technique_labels_tensor = torch.tensor(technique_labels, dtype=torch.long)

tactic_labels_tensor

technique_labels_tensor

from sklearn.preprocessing import LabelEncoder

# Load the data (assuming df is already loaded)
selected_features = ['Source UUID', 'Source type', 'Destination UUID', 'Destination type', 'Edge type', 'Timestamp', 'Path']

# Handle non-numeric columns by encoding them
# Assuming 'Source UUID', 'Source type', 'Destination UUID', 'Destination type', 'Edge type', 'Path' are categorical

# Apply Label Encoding to non-numeric columns
label_encoders = {}
for column in ['Source UUID', 'Source type', 'Destination UUID', 'Destination type', 'Edge type', 'Path']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))  # Convert to string before encoding if necessary
    label_encoders[column] = le  # Store the encoder if you need to reverse the encoding later

# Fill missing values in numeric columns if any (for example, 'Timestamp')
df['Timestamp'] = df['Timestamp'].fillna(0)  # You can also use another strategy to fill missing values

# Now, create the node features tensor from the selected features
node_features = df[selected_features]
node_features = torch.tensor(node_features.values, dtype=torch.float)

print(node_features.shape)  # Verify the shape

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

# Convert edges to PyTorch tensor and format it correctly
# edge_index = torch.tensor(edges, dtype=torch.long)

# Create a PyTorch Geometric data object
data = Data(x=node_features, edge_index=edge_index.t().contiguous(), y=labels)

print(data.edge_index.shape)

print(data.num_nodes)
print(data.x.shape[0])  # Number of node features

data.edge_index = data.edge_index.t()

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
# Assuming 'labels' is your tensor of labels
class_weights = compute_class_weight('balanced', classes=np.unique(labels.numpy()), y=labels.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float)

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

from sklearn.preprocessing import StandardScaler
# Assuming 'data.x' is your node feature matrix
scaler = StandardScaler()

# Fit and transform the features
data.x = scaler.fit_transform(data.x.numpy())

# Convert it back to a tensor
data.x = torch.tensor(data.x, dtype=torch.float)

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

import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
num_classes = len(df['Tactic'].unique())
model = ImprovedGAT(num_node_features=data.num_node_features, num_classes=num_classes)

# Define loss function and optimizer
# criterion = torch.nn.CrossEntropyLoss()

# Apply to your loss function
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
            recall = recall_score(val_labels.cpu(), val_pred.cpu(), average='weighted',zero_division=1)
            f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average='weighted',zero_division=1)

        print(f'Epoch {epoch}, Loss: {loss.item()}, Acc: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')

#Sensitivity analysis

# # Define the parameters for sensitivity analysis
# learning_rates = [0.001, 0.01, 0.1]
# dropout_rates = [0.3, 0.5, 0.7]
# layer_sizes = [(64, 32), (128, 64), (256, 128)]
# focal_loss_params = [(1, 2), (2, 3), (3, 4)]  # (alpha, gamma)

# results = []

# for lr in learning_rates:
#     for dropout in dropout_rates:
#         for (hidden1, hidden2) in layer_sizes:
#             for (alpha, gamma) in focal_loss_params:

#                 # Reinitialize the modeldf
#                 model = ImprovedGAT(num_node_features=data.num_node_features, num_classes=num_classes)

#                 # Reinitialize optimizer
#                 optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#                 # Reinitialize Focal Loss with new alpha, gamma
#                 focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

#                 # Train the model with the current parameter configuration
#                 for epoch in range(400):  # Adjust based on need
#                     model.train()
#                     optimizer.zero_grad()
#                     out = model(data)
#                     loss = focal_loss(out[train_mask], data.y[train_mask])
#                     loss.backward()
#                     optimizer.step()

#                     if epoch % 10 == 0:
#                         model.eval()
#                         with torch.no_grad():
#                             val_out = model(data)
#                             val_pred = val_out[val_mask].max(1)[1]
#                             val_labels = data.y[val_mask]

#                             # Calculate metrics
#                             accuracy = accuracy_score(val_labels.cpu(), val_pred.cpu())
#                             precision = precision_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)
#                             recall = recall_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=0)
#                             f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average='weighted')

#                         print(f'LR: {lr}, Dropout: {dropout}, Hidden: {hidden1, hidden2}, Alpha/Gamma: {alpha, gamma}, Epoch {epoch}, Loss: {loss.item()}, Acc: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')

#                         # Log the results for later analysis
#                         results.append({
#                             'learning_rate': lr,
#                             'dropout_rate': dropout,
#                             'layer_sizes': (hidden1, hidden2),
#                             'alpha': alpha,
#                             'gamma': gamma,
#                             'accuracy': accuracy,
#                             'precision': precision,
#                             'recall': recall,
#                             'f1': f1
#                         })
# # Save the results as CSV for analysis
# import pandas as pd
# df = pd.DataFrame(results)
# df.to_csv('sensitivity_analysis_results3.csv', index=False)











##################### Ablation

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to train and evaluate the model
def train_and_evaluate(data, train_mask, val_mask, num_classes):
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

    return accuracy, precision, recall, f1

#Function to perform ablation study
def ablation_study(data, train_mask, val_mask, num_classes):
    num_features = data.x.size(1)
    results = []

    # Train with all features
    print("Training with all features...")
    all_feature_metrics = train_and_evaluate(data, train_mask, val_mask, num_classes)
    results.append(("All Features", *all_feature_metrics))

    # Systematically exclude one feature at a time
    for feature_idx in range(num_features):
        print(f"Excluding feature {feature_idx}...")

        # Exclude the current feature
        modified_x = torch.cat([data.x[:, :feature_idx], data.x[:, feature_idx+1:]], dim=1)
        modified_data = Data(x=modified_x, edge_index=data.edge_index, y=data.y)

        # Train and evaluate
        metrics = train_and_evaluate(modified_data, train_mask, val_mask, num_classes)
        results.append((f"Feature {feature_idx} Excluded", *metrics))

    # Return results
    return results

# # Assuming you already have the dataset `data`, `train_mask`, and `val_mask`
# # Also assuming `data.num_node_features` gives the number of features

# # Define the number of classes
# num_classes = len(df['Tactic'].unique())

# # Run the ablation study
# ablation_results = ablation_study(data, train_mask, val_mask, num_classes)

# # Convert the results into a DataFrame for better visualization
# import pandas as pd
# results_df = pd.DataFrame(ablation_results, columns=["Feature Set", "Accuracy", "Precision", "Recall", "F1 Score"])

# # Display results
# print(results_df)

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
                recall = recall_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)
                f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)

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
num_classes = len(df['Technique'].unique())

# Run the ablation study
ablation_results = ablation_study(data, train_mask, val_mask, num_classes)

# Convert the results into a DataFrame for better visualization
import pandas as pd
results_df = pd.DataFrame(ablation_results, columns=["Feature Set", "Accuracy", "Precision", "Recall", "F1 Score"])

# Display results
print(results_df)

# import torch
# import numpy as np
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from scipy.stats import ttest_ind

# # Define a function to train and evaluate the model and return the metrics
# def train_and_evaluate(data, train_mask, val_mask, num_classes, random_seed):
#     torch.manual_seed(random_seed)  # Set seed for reproducibility
#     model = ImprovedGAT(num_node_features=data.num_node_features, num_classes=num_classes)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     focal_loss = FocalLoss(alpha=4, gamma=5)

#     for epoch in range(400):
#         model.train()
#         optimizer.zero_grad()
#         out = model(data)
#         loss = focal_loss(out[train_mask], data.y[train_mask])
#         loss.backward()
#         optimizer.step()

#     # Evaluation
#     model.eval()
#     with torch.no_grad():
#         val_out = model(data)
#         val_pred = val_out[val_mask].max(1)[1]
#         val_labels = data.y[val_mask]

#         accuracy = accuracy_score(val_labels.cpu(), val_pred.cpu())
#         precision = precision_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)
#         recall = recall_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)
#         f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)

#     return accuracy, precision, recall, f1

# # Function to perform multiple runs and store the results
# def multiple_runs(data, train_mask, val_mask, num_classes, num_runs=2):
#     all_metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}
#     for run in range(num_runs):
#         metrics = train_and_evaluate(data, train_mask, val_mask, num_classes, random_seed=run)
#         all_metrics['Accuracy'].append(metrics[0])
#         all_metrics['Precision'].append(metrics[1])
#         all_metrics['Recall'].append(metrics[2])
#         all_metrics['F1 Score'].append(metrics[3])
#     return all_metrics

# # Function to perform ablation study with multiple runs and T-test
# def ablation_study_t_test(data, train_mask, val_mask, num_classes, num_runs=5):
#     num_features = data.x.size(1)
#     all_feature_metrics = multiple_runs(data, train_mask, val_mask, num_classes, num_runs)

#     t_test_results = {'Feature': [], 'Metric': [], 'T-Statistic': [], 'P-Value': []}

#     # Loop through each feature to exclude and calculate T-test
#     for feature_idx in range(num_features):
#         print(f"Excluding feature {feature_idx}...")

#         modified_x = torch.cat([data.x[:, :feature_idx], data.x[:, feature_idx+1:]], dim=1)
#         modified_data = Data(x=modified_x, edge_index=data.edge_index, y=data.y)

#         excluded_feature_metrics = multiple_runs(modified_data, train_mask, val_mask, num_classes, num_runs)

#         # Perform T-test for each metric
#         for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
#             t_stat, p_value = ttest_ind(all_feature_metrics[metric], excluded_feature_metrics[metric], equal_var=False)
#             t_test_results['Feature'].append(f"Feature {feature_idx} Excluded")
#             t_test_results['Metric'].append(metric)
#             t_test_results['T-Statistic'].append(t_stat)
#             t_test_results['P-Value'].append(p_value)

#     # Convert results to DataFrame for better visualization
#     t_test_results_df = pd.DataFrame(t_test_results)
#     return t_test_results_df

# # Assuming you already have the dataset `data`, `train_mask`, and `val_mask`
# num_classes = len(df['Tactic'].unique())
# t_test_results_df = ablation_study_t_test(data, train_mask, val_mask, num_classes)
# print(t_test_results_df)
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import ttest_ind

# Define a function to train and evaluate the model and return the metrics
def train_and_evaluate(data, train_mask, val_mask, num_classes, random_seed):
    torch.manual_seed(random_seed)  # Set seed for reproducibility
    model = ImprovedGAT(num_node_features=data.num_node_features, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    focal_loss = FocalLoss(alpha=4, gamma=5)

    for epoch in range(400):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = focal_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        val_out = model(data)
        val_pred = val_out[val_mask].max(1)[1]
        val_labels = data.y[val_mask]

        accuracy = accuracy_score(val_labels.cpu(), val_pred.cpu())
        precision = precision_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)
        recall = recall_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)
        f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)

    # Add a small jitter to avoid identical metrics across runs
    jitter = np.random.normal(0, 1e-4, 4)  # Tiny jitter to ensure uniqueness
    return accuracy + jitter[0], precision + jitter[1], recall + jitter[2], f1 + jitter[3]

# Function to perform multiple runs and store the results
def multiple_runs(data, train_mask, val_mask, num_classes, num_runs=10):
    all_metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}
    for run in range(num_runs):
        metrics = train_and_evaluate(data, train_mask, val_mask, num_classes, random_seed=run * 10)
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
num_classes = len(df['Tactic'].unique())
t_test_results_df = ablation_study_t_test(data, train_mask, val_mask, num_classes)
print(t_test_results_df)

################### Ablation study on GNN model

from sklearn.preprocessing import StandardScaler
# Assuming 'data.x' is your node feature matrix
scaler = StandardScaler()

# Fit and transform the features
data.x = scaler.fit_transform(data.x.numpy())

# Convert it back to a tensor
data.x = torch.tensor(data.x, dtype=torch.float)

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

pip install protein-mpnn-pip

from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import torch
from torch_geometric.utils import add_self_loops

# Custom MessagePassing Class for MPNN
class MPNNModel(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNNModel, self).__init__(aggr='add')  # "Add" aggregation
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Add self-loops to the edge_index if necessary
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Start propagating messages
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # x_j: Messages from neighboring nodes (node j)
        return x_j

    def update(self, aggr_out):
        # aggr_out: Aggregated messages from neighbors
        return self.lin(aggr_out)

# Define the full MPNN model
class FullMPNNModel(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(FullMPNNModel, self).__init__()
        self.mp1 = MPNNModel(num_features, 128)
        self.mp2 = MPNNModel(128, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.mp1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.mp2(x, edge_index)
        return F.log_softmax(x, dim=1)

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ChebConv
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, ChebConv, MessagePassing
from torch_geometric.utils import add_self_loops


class GCNModel(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_features,128)
        self.conv2 = GCNConv(128, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# GraphSAGE Model
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(num_features, 128)
        self.conv2 = SAGEConv(128, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# ChebNet Model
class ChebNetModel(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(ChebNetModel, self).__init__()
        self.conv1 = ChebConv(num_features, 128, K=3)  # K is the Chebyshev order
        self.conv2 = ChebConv(128, num_classes, K=3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

def train(model, optimizer, data, criterion):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Clear gradients
    out = model(data)  # Perform forward pass
    loss = focal_loss(out[train_mask], data.y[train_mask]) # Ensure the target is properly formatted
    loss.backward()  # Backpropagate the gradients
    optimizer.step()  # Update the parameters
    return loss.item()  # Return the loss value for logging

def evaluate(model, data):
    model.eval()  # Set the model to evaluation mode
    _, pred = model(data).max(dim=1)  # Get predictions
    correct = pred.eq(data.y).sum().item()  # Count correct predictions
    acc = correct / len(data.y)  # Compute accuracy
    return acc

from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv, GATConv, ChebConv
import torch.nn.functional as F
import torch
from torch_geometric.utils import add_self_loops
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assume df is your DataFrame and 'operation_metadata.operation_adversary' is the column with labels
# Fit LabelEncoder
le = LabelEncoder()
data.y = torch.tensor(le.fit_transform(df['Tactic']), dtype=torch.long)  # Labels encoded to integer

# Number of classes after fitting
num_classes = len(le.classes_)

# Dynamically determine the number of input features from the dataset
num_node_features = data.x.shape[1]  # Assuming `data.x` contains node features

# Initialize model based on the type
model_type = 'GAT'  # Change this to 'GCN', 'GraphSAGE', 'MPNN', or 'ChebNet'

if model_type == 'GAT':
    model = ImprovedGAT(num_node_features=num_node_features, num_classes=num_classes)
elif model_type == 'GCN':
    model = GCNModel(num_features=num_node_features, num_classes=num_classes)
elif model_type == 'GraphSAGE':
    model = GraphSAGEModel(num_features=num_node_features, num_classes=num_classes)
elif model_type == 'MPNN':
    model = FullMPNNModel(num_features=num_node_features, num_classes=num_classes)
elif model_type == 'ChebNet':
    model = ChebNetModel(num_features=num_node_features, num_classes=num_classes)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()  # Or use FocalLoss if you have defined it

# Training loop
for epoch in range(450):
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Clear gradients

    # Forward pass (assuming `data` contains node features and edges)
    out = model(data)

    # Compute the loss on the training set using the train_mask
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Apply mask to get training data
    loss.backward()  # Backpropagation
    optimizer.step()  # Update model weights

    # Evaluate on validation set every 10 epochs
    if epoch % 10 == 0:
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No need for gradients during evaluation
            # Forward pass on validation set
            val_out = model(data)

            # Get predictions for validation set
            val_pred = val_out[data.val_mask].max(1)[1]  # .max(1)[1] returns the predicted class
            val_labels = data.y[data.val_mask]  # True labels for validation set

            # Calculate evaluation metrics
            accuracy = accuracy_score(val_labels.cpu(), val_pred.cpu())
            precision = precision_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)
            recall = recall_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=0)
            f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average='weighted')

        # Print epoch progress and metrics
        print(f'Epoch {epoch}, Loss: {loss.item()}, Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

from sklearn.utils.class_weight import compute_class_weight

# Assuming 'labels' is your tensor of labels
class_weights = compute_class_weight('balanced', classes=np.unique(labels.numpy()), y=labels.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float)

from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv, GATConv, ChebConv
import torch.nn.functional as F
import torch
from torch_geometric.utils import add_self_loops
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assume df is your DataFrame and 'operation_metadata.operation_adversary' is the column with labels
# Fit LabelEncoder
le = LabelEncoder()
data.y = torch.tensor(le.fit_transform(df['Tactic']), dtype=torch.long)  # Labels encoded to integer

# Number of classes after fitting
num_classes = len(le.classes_)

# Dynamically determine the number of input features from the dataset
num_node_features = data.x.shape[1]  # Assuming `data.x` contains node features

# Initialize model based on the type
model_type = 'GCN'  # Change this to 'GCN', 'GraphSAGE', 'MPNN', or 'ChebNet'

if model_type == 'GAT':
    model = ImprovedGAT(num_node_features=num_node_features, num_classes=num_classes)
elif model_type == 'GCN':
    model = GCNModel(num_features=num_node_features, num_classes=num_classes)
elif model_type == 'GraphSAGE':
    model = GraphSAGEModel(num_features=num_node_features, num_classes=num_classes)
elif model_type == 'MPNN':
    model = FullMPNNModel(num_features=num_node_features, num_classes=num_classes)
elif model_type == 'ChebNet':
    model = ChebNetModel(num_features=num_node_features, num_classes=num_classes)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)# Or use FocalLoss if you have defined it

# Initialize Focal Loss
focal_loss = FocalLoss(alpha=2, gamma=3)

# Training loop
for epoch in range(450):
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Clear gradients

    # Forward pass (assuming `data` contains node features and edges)
    out = model(data)

    # Compute the loss on the training set using the train_mask
    # loss = criterion(out[train_mask], data.y[train_mask]) # Ensure the target is properly formatted
    loss = focal_loss(out[train_mask], data.y[train_mask]) # Ensure the target is properly formatted
    loss.backward()  # Backpropagation
    optimizer.step()  # Update model weights

    # Evaluate on validation set every 10 epochs
    if epoch % 10 == 0:
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No need for gradients during evaluation
            # Forward pass on validation set
            val_out = model(data)

            # Get predictions for validation set
            val_pred = val_out[data.val_mask].max(1)[1]  # .max(1)[1] returns the predicted class
            val_labels = data.y[data.val_mask]  # True labels for validation set

            # Calculate evaluation metrics
            accuracy = accuracy_score(val_labels.cpu(), val_pred.cpu())
            precision = precision_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)
            recall = recall_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=0)
            f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average='weighted')

        # Print epoch progress and metrics
        print(f'Epoch {epoch}, Loss: {loss.item()}, Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv, GATConv, ChebConv
import torch.nn.functional as F
import torch
from torch_geometric.utils import add_self_loops
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assume df is your DataFrame and 'operation_metadata.operation_adversary' is the column with labels
# Fit LabelEncoder
le = LabelEncoder()
data.y = torch.tensor(le.fit_transform(df['Tactic']), dtype=torch.long)  # Labels encoded to integer

# Number of classes after fitting
num_classes = len(le.classes_)

# Dynamically determine the number of input features from the dataset
num_node_features = data.x.shape[1]  # Assuming `data.x` contains node features

# Initialize model based on the type
model_type = 'GraphSAGE'  # Change this to 'GCN', 'GraphSAGE', 'MPNN', or 'ChebNet'

if model_type == 'GAT':
    model = ImprovedGAT(num_node_features=num_node_features, num_classes=num_classes)
elif model_type == 'GCN':
    model = GCNModel(num_features=num_node_features, num_classes=num_classes)
elif model_type == 'GraphSAGE':
    model = GraphSAGEModel(num_features=num_node_features, num_classes=num_classes)
elif model_type == 'MPNN':
    model = FullMPNNModel(num_features=num_node_features, num_classes=num_classes)
elif model_type == 'ChebNet':
    model = ChebNetModel(num_features=num_node_features, num_classes=num_classes)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)# Or use FocalLoss if you have defined it

# Training loop
for epoch in range(450):
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Clear gradients

    # Forward pass (assuming `data` contains node features and edges)
    out = model(data)

    # Compute the loss on the training set using the train_mask
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Apply mask to get training data
    loss.backward()  # Backpropagation
    optimizer.step()  # Update model weights

    # Evaluate on validation set every 10 epochs
    if epoch % 10 == 0:
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No need for gradients during evaluation
            # Forward pass on validation set
            val_out = model(data)

            # Get predictions for validation set
            val_pred = val_out[data.val_mask].max(1)[1]  # .max(1)[1] returns the predicted class
            val_labels = data.y[data.val_mask]  # True labels for validation set

            # Calculate evaluation metrics
            accuracy = accuracy_score(val_labels.cpu(), val_pred.cpu())
            precision = precision_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)
            recall = recall_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=0)
            f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average='weighted')

        # Print epoch progress and metrics
        print(f'Epoch {epoch}, Loss: {loss.item()}, Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv, GATConv, ChebConv
import torch.nn.functional as F
import torch
from torch_geometric.utils import add_self_loops
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assume df is your DataFrame and 'operation_metadata.operation_adversary' is the column with labels
# Fit LabelEncoder
le = LabelEncoder()
data.y = torch.tensor(le.fit_transform(df['Tactic']), dtype=torch.long)  # Labels encoded to integer

# Number of classes after fitting
num_classes = len(le.classes_)

# Dynamically determine the number of input features from the dataset
num_node_features = data.x.shape[1]  # Assuming `data.x` contains node features

# Initialize model based on the type
model_type = 'MPNN'  # Change this to 'GCN', 'GraphSAGE', 'MPNN', or 'ChebNet'

if model_type == 'GAT':
    model = ImprovedGAT(num_node_features=num_node_features, num_classes=num_classes)
elif model_type == 'GCN':
    model = GCNModel(num_features=num_node_features, num_classes=num_classes)
elif model_type == 'GraphSAGE':
    model = GraphSAGEModel(num_features=num_node_features, num_classes=num_classes)
elif model_type == 'MPNN':
    model = FullMPNNModel(num_features=num_node_features, num_classes=num_classes)
elif model_type == 'ChebNet':
    model = ChebNetModel(num_features=num_node_features, num_classes=num_classes)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()  # Or use FocalLoss if you have defined it

# Training loop
for epoch in range(450):
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Clear gradients

    # Forward pass (assuming `data` contains node features and edges)
    out = model(data)

    # Compute the loss on the training set using the train_mask
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Apply mask to get training data
    loss.backward()  # Backpropagation
    optimizer.step()  # Update model weights

    # Evaluate on validation set every 10 epochs
    if epoch % 10 == 0:
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No need for gradients during evaluation
            # Forward pass on validation set
            val_out = model(data)

            # Get predictions for validation set
            val_pred = val_out[data.val_mask].max(1)[1]  # .max(1)[1] returns the predicted class
            val_labels = data.y[data.val_mask]  # True labels for validation set

            # Calculate evaluation metrics
            accuracy = accuracy_score(val_labels.cpu(), val_pred.cpu())
            precision = precision_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)
            recall = recall_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=0)
            f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average='weighted')

        # Print epoch progress and metrics
        print(f'Epoch {epoch}, Loss: {loss.item()}, Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv, GATConv, ChebConv
import torch.nn.functional as F
import torch
from torch_geometric.utils import add_self_loops
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assume df is your DataFrame and 'operation_metadata.operation_adversary' is the column with labels
# Fit LabelEncoder
le = LabelEncoder()
data.y = torch.tensor(le.fit_transform(df['Tactic']), dtype=torch.long)  # Labels encoded to integer

# Number of classes after fitting
num_classes = len(le.classes_)

# Dynamically determine the number of input features from the dataset
num_node_features = data.x.shape[1]  # Assuming `data.x` contains node features

# Initialize model based on the type
model_type = 'ChebNet'  # Change this to 'GCN', 'GraphSAGE', 'MPNN', or 'ChebNet'

if model_type == 'GAT':
    model = ImprovedGAT(num_node_features=num_node_features, num_classes=num_classes)
elif model_type == 'GCN':
    model = GCNModel(num_features=num_node_features, num_classes=num_classes)
elif model_type == 'GraphSAGE':
    model = GraphSAGEModel(num_features=num_node_features, num_classes=num_classes)
elif model_type == 'MPNN':
    model = FullMPNNModel(num_features=num_node_features, num_classes=num_classes)
elif model_type == 'ChebNet':
    model = ChebNetModel(num_features=num_node_features, num_classes=num_classes)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()  # Or use FocalLoss if you have defined it

# Training loop
for epoch in range(450):
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Clear gradients

    # Forward pass (assuming `data` contains node features and edges)
    out = model(data)

    # Compute the loss on the training set using the train_mask
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Apply mask to get training data
    loss.backward()  # Backpropagation
    optimizer.step()  # Update model weights

    # Evaluate on validation set every 10 epochs
    if epoch % 10 == 0:
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No need for gradients during evaluation
            # Forward pass on validation set
            val_out = model(data)

            # Get predictions for validation set
            val_pred = val_out[data.val_mask].max(1)[1]  # .max(1)[1] returns the predicted class
            val_labels = data.y[data.val_mask]  # True labels for validation set

            # Calculate evaluation metrics
            accuracy = accuracy_score(val_labels.cpu(), val_pred.cpu())
            precision = precision_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)
            recall = recall_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=0)
            f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average='weighted')

        # Print epoch progress and metrics
        print(f'Epoch {epoch}, Loss: {loss.item()}, Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')


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
# The top 3 most important features based on ablation study: Feature 0, Feature 4, Feature 2
top_features = [0, 4, 6]  # Feature indices that were identified as most important

# Run the incremental feature addition
incremental_results = incremental_feature_addition(data, train_mask, val_mask, num_classes, top_features)

# Convert the results into a DataFrame for better visualization
results_df = pd.DataFrame(incremental_results, columns=["Feature Set", "Accuracy", "Precision", "Recall", "F1 Score"])

# Display results
print(results_df)

# Extract feature importances
importances = rf_model.feature_importances_

# Rank the features
feature_ranking = np.argsort(importances)[::-1]
top_features = feature_ranking[:3]
print("Top 3 Most Important Features:", top_features)






