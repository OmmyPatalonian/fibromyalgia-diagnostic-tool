#fibromyalgia-diagnostic-tool/gan-cnn-project/src/gnn/gnn_model.py
import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        print("Initializing GNN...")
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        print("GNN initialized.")

    def forward(self, data):
        print("Forward pass through GNN...")
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, data.batch)
        output = self.fc(x)
        print("Forward pass completed.")
        return output