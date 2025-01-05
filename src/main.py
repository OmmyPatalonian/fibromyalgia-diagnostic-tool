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

    # Split data
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)

    # Step 3: Balance training data
    majority_class = train_data[train_data['Condition'] == 1]
    minority_class = train_data[train_data['Condition'] == 0]
    if minority_class.empty:
        print("[ERROR] Minority class is empty. Check preprocessing.")
        return
    minority_class_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
    balanced_train_data = pd.concat([majority_class, minority_class_upsampled])

    print("[DEBUG] Balanced train class distribution:", balanced_train_data['Condition'].value_counts())

    # Step 4: GAN Training
    latent_dim = 100
    generator = build_generator(latent_dim, 3)
    discriminator = build_discriminator(3)
    train_gan(generator, discriminator, balanced_train_data.drop(columns=['Condition']).values, epochs=100, batch_size=64, latent_dim=latent_dim)

    # Generate synthetic data
    noise = torch.randn((len(balanced_train_data), latent_dim))
    synthetic_data = generator(noise).detach().numpy()
    synthetic_data_df = pd.DataFrame(synthetic_data, columns=balanced_train_data.columns.drop('Condition'))
    synthetic_data_df['Condition'] = 1  # Label synthetic data as class 1

    # Combine real and synthetic data
    combined_data = pd.concat([balanced_train_data, synthetic_data_df])

    # Step 5: Train GNN
    gnn_model = GNN(model_type='GCN', input_dim=3, hidden_dim=16, output_dim=64, pooling=True)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(100):
        gnn_model.train()
        epoch_loss = 0
        for i in range(0, len(combined_data), 64):
            batch_data = combined_data.iloc[i:i + 64]
            # Create batch
            batch_data_list = [
                Data(x=torch.tensor(row[['HRV', 'GSR', 'EMG']].values, dtype=torch.float).unsqueeze(0),
                     y=torch.tensor(row['Condition'], dtype=torch.float).unsqueeze(0),
                     edge_index=create_graph_edges(torch.tensor(row[['HRV', 'GSR', 'EMG']].values, dtype=torch.float).unsqueeze(0)))
                for _, row in batch_data.iterrows()
            ]
            batch_graph = Batch.from_data_list(batch_data_list)

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
        print(f"Epoch {epoch+1}, GNN Loss: {epoch_loss / (len(combined_data) // 64)}")

    # Prepare gnn_output for 2D CNN
    gnn_output = gnn_model(batch_graph)
    print(f"[DEBUG] GNN raw output shape: {gnn_output.shape}")  # Debugging GNN output

    # Ensure GNN output has correct dimensions
    gnn_output = gnn_output.view(gnn_output.size(0), -1)  # Flatten if needed
    gnn_output = gnn_output.unsqueeze(1).expand(-1, 3, -1, -1)  # Expand to [batch_size, 3, height, width]

    # Resize for CNN input
    gnn_output_resized = F.interpolate(gnn_output, size=(224, 224), mode='bilinear', align_corners=False)
    print(f"[DEBUG] GNN output after interpolate: {gnn_output_resized.shape}")

    # Verify the shape
    print(f"[DEBUG] Final GNN output shape (before CNN): {gnn_output_resized.shape}")  # Should be [batch_size, 3, 224, 224]

    # Pass the prepared output to the CNN model
    cnn_model = CNN2D(input_channels=3, output_dim=1, dropout_rate=0.3, weight_decay=1e-4)
    print(f"[DEBUG] Input to CNN shape: {gnn_output_resized.shape}")  # Debugging before CNN forward pass
    cnn_output = cnn_model(gnn_output_resized)
    print(f"[DEBUG] CNN output shape: {cnn_output.shape}")

    # Evaluate on test set
    gnn_model.eval()
    cnn_model.eval()

    # Create dataset
    test_dataset = FibroDataset(test_data)

    # Create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    all_cnn_predictions = []
    all_gnn_predictions = []
    all_test_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            # Prepare features and labels
            test_features = features  # Shape: [batch_size, 3]
            test_labels = labels.unsqueeze(1)  # Shape: [batch_size, 1]

            # Process GNN output
            test_graph_data_list = [
                Data(x=feature.unsqueeze(0),
                     y=label.unsqueeze(0),
                     edge_index=create_graph_edges(feature.unsqueeze(0)))
                for feature, label in zip(test_features, test_labels)
            ]
            test_graph_batch = Batch.from_data_list(test_graph_data_list)

            gnn_output = gnn_model(test_graph_batch)
            gnn_output = gnn_output.mean(dim=1)  # Aggregate predictions
            gnn_predictions = torch.round(torch.sigmoid(gnn_output))

            # Process CNN output
            print(f"[DEBUG] gnn_output shape before reshape: {gnn_output.shape}")
            gnn_output_reshaped = gnn_output.view(gnn_output.size(0), 1, 1, 1).expand(-1, 3, 64, 64)
            print(f"[DEBUG] Reshaped gnn_output for CNN input: {gnn_output_reshaped.shape}")

            gnn_output_resized = F.interpolate(gnn_output_reshaped, size=(224, 224), mode='bilinear', align_corners=False)
            cnn_output = cnn_model(gnn_output_resized)
            cnn_predictions = torch.round(torch.sigmoid(cnn_output.squeeze()))

            # Collect results
            all_cnn_predictions.append(cnn_predictions.cpu())
            all_gnn_predictions.append(gnn_predictions.cpu())
            all_test_labels.append(test_labels.cpu())

    # Combine predictions and labels
    all_cnn_predictions = torch.cat(all_cnn_predictions).numpy()
    all_gnn_predictions = torch.cat(all_gnn_predictions).numpy()
    all_test_labels = torch.cat(all_test_labels).numpy()

    # Evaluate models
    print(f"[DEBUG] all_test_labels shape: {all_test_labels.shape}")
    print(f"[DEBUG] all_cnn_predictions shape: {all_cnn_predictions.shape}")
    print(f"[DEBUG] all_gnn_predictions shape: {all_gnn_predictions.shape}")

    gnn_precision, gnn_recall, gnn_f1, gnn_roc_auc = evaluate_model(all_test_labels, all_gnn_predictions)
    cnn_precision, cnn_recall, cnn_f1, cnn_roc_auc = evaluate_model(all_test_labels, all_cnn_predictions)

    print(f"GNN Precision: {gnn_precision}, Recall: {gnn_recall}, F1: {gnn_f1}, ROC AUC: {gnn_roc_auc}")
    print(f"CNN Precision: {cnn_precision}, Recall: {cnn_recall}, F1: {cnn_f1}, ROC AUC: {cnn_roc_auc}")

if __name__ == "__main__":
    main()