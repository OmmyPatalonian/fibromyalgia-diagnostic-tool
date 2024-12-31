# main.py

import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from gan.generator import build_generator
from gan.discriminator import build_discriminator
from gan.gan import build_gan
from gan.train_gan import train_gan
from data.generate_data import generate_realistic_data
from data.save_csv import generate_and_save_csv
from gnn.gnn_model import GNN
from cnn.fibro_cnn import FibroCNN
from cnn.analyze_and_report import analyze_and_report
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import seaborn as sns

def create_graph_edges(features):
    num_nodes = features.size(0)
    edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long).t().contiguous()
    return edge_index

if __name__ == "__main__":
    print("Starting main script...")
    latent_dim = 10
    data_dim = 3
    epochs = 1000
    num_samples = 1000
    num_files = 20

    # Prepare GAN
    print("Preparing GAN...")
    generator = build_generator(latent_dim, data_dim, num_classes=2)
    discriminator = build_discriminator(data_dim, num_classes=2)
    gan = build_gan(generator, discriminator)

    # Compile models
    generator.compile(optimizer='adam', loss='binary_crossentropy')
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    fibro_data = generate_realistic_data(num_samples, is_fibro=True)
    normal_data = generate_realistic_data(num_samples, is_fibro=False)

    # Visualize the distributions before scaling
    plt.figure(figsize=(12, 6))
    sns.histplot(fibro_data.flatten(), kde=True, label='Fibro Data (Before Scaling)', color='blue')
    sns.histplot(normal_data.flatten(), kde=True, label='Normal Data (Before Scaling)', color='orange')
    plt.legend()
    plt.title('Feature Distributions Before Scaling')
    plt.show()

    # Separate scaling for each class
    scaler_fibro = MinMaxScaler()
    scaler_normal = MinMaxScaler()
    fibro_data = scaler_fibro.fit_transform(fibro_data)
    normal_data = scaler_normal.fit_transform(normal_data)

    # Visualize the distributions after scaling
    plt.figure(figsize=(12, 6))
    sns.histplot(fibro_data.flatten(), kde=True, label='Fibro Data (After Scaling)', color='blue')
    sns.histplot(normal_data.flatten(), kde=True, label='Normal Data (After Scaling)', color='orange')
    plt.legend()
    plt.title('Feature Distributions After Scaling')
    plt.show()

    # Combine the scaled data
    training_data = np.vstack([fibro_data, normal_data])

    # Balance the dataset
    from sklearn.utils import resample

    # Separate minority and majority classes
    fibro = pd.DataFrame(training_data[training_data[:, -1] == 1])
    normal = pd.DataFrame(training_data[training_data[:, -1] == 0])

    # Upsample minority class
    fibro_upsampled = resample(fibro, replace=True, n_samples=len(normal), random_state=42)

    # Combine majority class with upsampled minority class
    training_data_balanced = np.vstack([normal, fibro_upsampled])

    # Debug class distribution
    print("Class Distribution in Training Data:")
    print(pd.DataFrame(training_data_balanced).groupby(2).size())  # Assuming the last column is the label

    train_gan(generator, discriminator, gan, training_data_balanced, epochs, 32, latent_dim)
    generate_and_save_csv(generator, scaler_fibro, latent_dim, num_samples, num_files)

    # Load models
    print("Loading models...")
    gnn_model = GNN(input_dim=data_dim, hidden_dim=64, output_dim=32)
    cnn_model = FibroCNN(input_dim=32, pos_weight=torch.tensor([5.0]))  # Adjust pos_weight based on imbalance

    # Process Generated CSVs
    print("Processing generated CSVs...")
    for i, csv_file in enumerate(os.listdir("generated_csvs")):
        filepath = os.path.join("generated_csvs", csv_file)
        data = pd.read_csv(filepath)

        # Drop rows with NaN in Condition
        data = data.dropna(subset=['Condition'])

        # Or fill NaN with a default value if appropriate
        data['Condition'] = data['Condition'].fillna(0)  # Assuming 0 means "Normal"

        data['Condition'] = data['Condition'].map({'fibro': 1, 'normal': 0})  # Encode labels
        condition_label = data['Condition'][0]  # Extract condition from file

        # Debug a sample
        print("CSV Data Sample:")
        print(data.head())
        print("Labels Sample:")
        print(data['Condition'].head())

        # GNN processing (convert to graph, extract features)
        features = torch.tensor(data[['HRV', 'GSR', 'EMG']].values, dtype=torch.float)
        edge_index = create_graph_edges(features)  # Define graph creation logic
        graph_data = Data(x=features, edge_index=edge_index)
        gnn_features = gnn_model(graph_data)

        # Analyze with CNN and report
        analyze_and_report(cnn_model, gnn_features, [0.6, 0.8, 0.5], condition_label)
    print("Main script completed.")