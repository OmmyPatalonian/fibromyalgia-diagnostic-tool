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

if __name__ == "__main__":
    print("Starting main script...")
    latent_dim = 10
    data_dim = 3
    epochs = 1000
    num_samples = 1000
    num_files = 20

    # Prepare GAN
    print("Preparing GAN...")
    generator = build_generator(latent_dim, data_dim)
    discriminator = build_discriminator(data_dim)
    gan = build_gan(generator, discriminator)

    fibro_data = generate_realistic_data(num_samples, is_fibro=True)
    normal_data = generate_realistic_data(num_samples, is_fibro=False)
    training_data = np.vstack([fibro_data, normal_data])
    scaler = MinMaxScaler((0, 1))
    training_data = scaler.fit_transform(training_data)

    train_gan(generator, discriminator, gan, training_data, epochs, 32, latent_dim)
    generate_and_save_csv(generator, scaler, latent_dim, num_samples, num_files)

    # Load models
    print("Loading models...")
    gnn_model = GNN(input_dim=data_dim, hidden_dim=64, output_dim=32)
    cnn_model = FibroCNN(input_dim=32)

    # Process Generated CSVs
    print("Processing generated CSVs...")
    for i, csv_file in enumerate(os.listdir("generated_csvs")):
        filepath = os.path.join("generated_csvs", csv_file)
        data = pd.read_csv(filepath)
        condition_label = data['Condition'][0]  # Extract condition from file

        # GNN processing (convert to graph, extract features)
        features = torch.tensor(data[['HRV', 'GSR', 'EMG']].values, dtype=torch.float)
        edge_index = create_graph_edges(features)  # Define graph creation logic
        graph_data = Data(x=features, edge_index=edge_index)
        gnn_features = gnn_model(graph_data)

        # Analyze with CNN and report
        analyze_and_report(cnn_model, gnn_features, [0.6, 0.8, 0.5], condition_label)
    print("Main script completed.")