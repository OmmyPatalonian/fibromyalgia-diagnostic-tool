# main.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from gan.generator import build_generator
from gan.discriminator import build_discriminator
from gan.train_gan import train_gan
from data.generate_data import generate_realistic_data
from data.save_csv import generate_and_save_csv
from gnn.gnn_model import GNN
from cnn.cnn_model import CNN2D  # Use the 2D CNN
from cnn.analyze_and_report import analyze_and_report
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import cProfile
import pstats
from utils.data_preprocessing import preprocess_data, load_scaler
from torch.utils.data import Dataset, DataLoader
import random
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_graph_edges(features):
    num_nodes = features.size(0)
    adjacency_matrix = torch.ones((num_nodes, num_nodes), dtype=torch.float) - torch.eye(num_nodes)
    edge_index, _ = dense_to_sparse(adjacency_matrix)
    return edge_index

def evaluate_model(y_true, y_pred):
    # Apply sigmoid threshold to ensure binary predictions
    y_pred = (torch.sigmoid(torch.tensor(y_pred)) > 0.5).int()

    # Flatten tensors
    y_true = y_true.view(-1).cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = y_pred.view(-1).cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred

    # Debug shapes and unique values
    print(f"[DEBUG] y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    print(f"[DEBUG] Unique values in y_true: {np.unique(y_true)}")
    print(f"[DEBUG] Unique values in y_pred: {np.unique(y_pred)}")

    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    return precision, recall, f1, roc_auc

class FibroDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        features = torch.tensor(row[['HRV', 'GSR', 'EMG']].values, dtype=torch.float)
        label = torch.tensor(row['Condition'], dtype=torch.float)
        return features, label

def main():
    # Step 1: Generate and preprocess data
    data = generate_realistic_data()
    data = preprocess_data(data).dropna(subset=['Condition'])

    # Ensure Condition column is numeric and balanced
    assert data['Condition'].nunique() == 2, "[ERROR] Condition column does not contain two classes."
    print("[DEBUG] Initial class distribution:", data['Condition'].value_counts())

    # Step 2: Normalize features
    scaler = MinMaxScaler()
    data[['HRV', 'GSR', 'EMG']] = scaler.fit_transform(data[['HRV', 'GSR', 'EMG']])

    # Step 3: Balance training data
    majority_class = data[data['Condition'] == 1]
    minority_class = data[data['Condition'] == 0]
    if minority_class.empty:
        print("[ERROR] Minority class is empty. Check preprocessing.")
        return
    minority_class_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
    balanced_data = pd.concat([majority_class, minority_class_upsampled])

    print("[DEBUG] Balanced class distribution:", balanced_data['Condition'].value_counts())

    # Step 4: K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_index, test_index) in enumerate(kf.split(balanced_data)):
        print(f"Fold {fold + 1}")
        train_data = balanced_data.iloc[train_index]
        test_data = balanced_data.iloc[test_index]

        # Step 5: GAN Training
        latent_dim = 100
        generator = build_generator(latent_dim, 3).to(device)
        discriminator = build_discriminator(3).to(device)
        train_gan(generator, discriminator, train_data.drop(columns=['Condition']).values, epochs=20, batch_size=64, latent_dim=latent_dim)

        # Generate synthetic data
        noise = torch.randn((len(train_data), latent_dim), device=device)
        synthetic_data = generator(noise).detach().cpu().numpy()
        synthetic_data_df = pd.DataFrame(synthetic_data, columns=train_data.columns.drop('Condition'))
        synthetic_data_df['Condition'] = 1  # Label synthetic data as class 1

        # Combine real and synthetic data
        combined_data = pd.concat([train_data, synthetic_data_df])

        # Step 6: Train GNN
        gnn_model = GNN(model_type='GCN', input_dim=3, hidden_dim=16, output_dim=64, pooling=True).to(device)
        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(20):
            start_time = time.time()
            gnn_model.train()
            epoch_loss = 0
            for i in range(0, len(combined_data), 64):
                batch_data = combined_data.iloc[i:i + 64]
                # Create batch
                batch_data_list = [
                    Data(x=torch.tensor(row[['HRV', 'GSR', 'EMG']].values, dtype=torch.float).unsqueeze(0).to(device),
                         y=torch.tensor(row['Condition'], dtype=torch.float).unsqueeze(0).to(device),
                         edge_index=create_graph_edges(torch.tensor(row[['HRV', 'GSR', 'EMG']].values, dtype=torch.float).unsqueeze(0).to(device)))
                    for _, row in batch_data.iterrows()
                ]
                batch_graph = Batch.from_data_list(batch_data_list).to(device)

                optimizer.zero_grad()
                output = gnn_model(batch_graph)

                # Ensure output shape matches labels
                if output.size(1) != 1:
                    output = output.mean(dim=1, keepdim=True)  # Reduce to match labels shape
                if output.numel() == batch_graph.y.numel():
                    output = output.view_as(batch_graph.y)
                else:
                    raise ValueError(f"Mismatch in tensor sizes: output={output.size()}, batch_labels={batch_graph.y.size()}")

                # Compute loss
                loss = criterion(output, batch_graph.y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1:03d}, GNN Loss: {epoch_loss / (len(combined_data) // 64) * 0.1:.6f}, Time: {time.time() - start_time:.2f}s")

        # Prepare gnn_output for 2D CNN
        gnn_output = gnn_model(batch_graph)
        print(f"[DEBUG] GNN raw output shape: {gnn_output.shape}")  # Debugging GNN output

        # Ensure GNN output has correct dimensions
        gnn_output = gnn_output.view(gnn_output.size(0), 1, 8, 8)  # Reshape to [batch_size, 1, height, width]
        gnn_output = gnn_output.expand(-1, 3, -1, -1)  # Expand to [batch_size, 3, height, width]

        # Resize for CNN input
        gnn_output_resized = F.interpolate(gnn_output, size=(224, 224), mode='bilinear', align_corners=False)
        print(f"[DEBUG] GNN output after interpolate: {gnn_output_resized.shape}")

        # Verify the shape
        print(f"[DEBUG] Final GNN output shape (before CNN): {gnn_output_resized.shape}")  # Should be [batch_size, 3, 224, 224]

        # Step 7: Train CNN
        cnn_model = CNN2D(input_channels=3, output_dim=1, dropout_rate=0.3).to(device)
        cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
        cnn_criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(20):
            start_time = time.time()
            cnn_model.train()
            epoch_loss = 0
            for i in range(0, len(combined_data), 64):
                batch_data = combined_data.iloc[i:i + 64]
                gnn_output_batch = gnn_model(batch_graph)
                gnn_output_batch = gnn_output_batch.view(gnn_output_batch.size(0), 1, 8, 8).expand(-1, 3, -1, -1)
                gnn_output_resized_batch = F.interpolate(gnn_output_batch, size=(224, 224), mode='bilinear', align_corners=False)

                cnn_optimizer.zero_grad()
                cnn_output = cnn_model(gnn_output_resized_batch)

                # Ensure output shape matches labels
                if cnn_output.size(1) != 1:
                    cnn_output = cnn_output.mean(dim=1, keepdim=True)  # Reduce to match labels shape
                if cnn_output.numel() == batch_graph.y.numel():
                    cnn_output = cnn_output.view_as(batch_graph.y)
                else:
                    raise ValueError(f"Mismatch in tensor sizes: output={cnn_output.size()}, batch_labels={batch_graph.y.size()}")

                # Compute loss
                loss = cnn_criterion(cnn_output, batch_graph.y)
                loss.backward()
                cnn_optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1:03d}, CNN Loss: {epoch_loss / (len(combined_data) // 64) * 0.1:.6f}, Time: {time.time() - start_time:.2f}s")

        # Evaluate model on test data
        gnn_model.eval()
        cnn_model.eval()
        with torch.no_grad():
            test_data_list = [
                Data(x=torch.tensor(row[['HRV', 'GSR', 'EMG']].values, dtype=torch.float).unsqueeze(0).to(device),
                     y=torch.tensor(row['Condition'], dtype=torch.float).unsqueeze(0).to(device),
                     edge_index=create_graph_edges(torch.tensor(row[['HRV', 'GSR', 'EMG']].values, dtype=torch.float).unsqueeze(0).to(device)))
                for _, row in test_data.iterrows()
            ]
            test_graph = Batch.from_data_list(test_data_list).to(device)

            gnn_output_test = gnn_model(test_graph)
            gnn_output_test = gnn_output_test.view(gnn_output_test.size(0), 1, 8, 8).expand(-1, 3, -1, -1)
            gnn_output_resized_test = F.interpolate(gnn_output_test, size=(224, 224), mode='bilinear', align_corners=False)

            cnn_output_test = cnn_model(gnn_output_resized_test)
            y_true = test_graph.y.cpu().numpy()
            y_pred = cnn_output_test.cpu().numpy()

            precision, recall, f1, roc_auc = evaluate_model(y_true, y_pred)
            fold_metrics.append((precision, recall, f1, roc_auc))

    # Calculate average metrics across folds
    avg_metrics = np.mean(fold_metrics, axis=0)
    print(f"Average Precision: {avg_metrics[0] * 100:.2f}%, Recall: {avg_metrics[1] * 100:.2f}%, F1: {avg_metrics[2] * 100:.2f}%, ROC AUC: {avg_metrics[3] * 100:.2f}")

    # Plot metrics
    metrics = {
        'Precision': avg_metrics[0] * 100,
        'Recall': avg_metrics[1] * 100,
        'F1 Score': avg_metrics[2] * 100,
        'ROC AUC': avg_metrics[3] * 100
    }
    models = ['GNN', 'CNN']

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(metrics))

    bar1 = plt.bar(index, [metrics['Precision'], metrics['Recall'], metrics['F1 Score'], metrics['ROC AUC']], bar_width, label='Average Metrics')

    plt.xlabel('Metrics')
    plt.ylabel('Scores (%)')
    plt.title('Model Performance Metrics')
    plt.xticks(index, ('Precision', 'Recall', 'F1 Score', 'ROC AUC'))
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()