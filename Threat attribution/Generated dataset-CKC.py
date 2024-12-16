#Ablation Study #####

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to train and evaluate the model
def train_and_evaluate(data, train_mask, val_mask, num_classes):
    # Instantiate the model
    model = ImprovedGAT(num_node_features=data.num_node_features, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    focal_loss = FocalLoss(alpha=4, gamma=5)

    # Training loop
    for epoch in range(400):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = focal_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Print metrics every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(data)
                val_pred = val_out[val_mask].max(1)[1]
                val_labels = data.y[val_mask]

                # Calculate metrics
                accuracy = accuracy_score(val_labels.cpu(), val_pred.cpu())
                precision = precision_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)
                recall = recall_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)
                f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=1)

    # Return final metrics
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

from scipy.stats import ttest_ind
import pandas as pd

# Assuming 'results_df' is your DataFrame
all_features_metrics = results_df.loc[results_df['Feature Set'] == "All Features", ['Accuracy', 'Precision', 'Recall', 'F1 Score']].values[0]

# Dictionary to store T-test results
t_test_results = {'Feature': [], 'Metric': [], 'T-Statistic': [], 'P-Value': []}

for index, row in results_df.iterrows():
    if row['Feature Set'] != "All Features":
        feature_name = row['Feature Set']

        for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
            # Perform T-test between 'All Features' metrics and each feature excluded
            t_stat, p_value = ttest_ind([all_features_metrics[metric]], [row[metric]], equal_var=False)
            t_test_results['Feature'].append(feature_name)
            t_test_results['Metric'].append(metric)
            t_test_results['T-Statistic'].append(t_stat)
            t_test_results['P-Value'].append(p_value)

# Convert results to a DataFrame for easy viewing
t_test_results_df = pd.DataFrame(t_test_results)
print(t_test_results_df)

# Dictionary to store percentage difference results
percentage_diff_results = {'Feature Set': [], 'Metric': [], 'Percentage Difference': []}

# Get baseline metrics from "All Features" row
baseline_metrics = results_df[results_df['Feature Set'] == "All Features"].iloc[0, 1:]

# Loop through each excluded feature and calculate percentage differences
for index, row in results_df.iterrows():
    if row['Feature Set'] != "All Features":
        feature_name = row['Feature Set']

        for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
            # Calculate percentage difference
            percentage_diff = ((row[metric] - baseline_metrics[metric]) / baseline_metrics[metric]) * 100
            percentage_diff_results['Feature Set'].append(feature_name)
            percentage_diff_results['Metric'].append(metric)
            percentage_diff_results['Percentage Difference'].append(percentage_diff)

# Convert to DataFrame for easier viewing
percentage_diff_df = pd.DataFrame(percentage_diff_results)
print(percentage_diff_df)

import torch
import numpy as np
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

    return accuracy, precision, recall, f1

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


num_classes = len(df['operation_metadata.operation_adversary'].unique())
t_test_results_df = ablation_study_t_test(data, train_mask, val_mask, num_classes)
print(t_test_results_df)

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


