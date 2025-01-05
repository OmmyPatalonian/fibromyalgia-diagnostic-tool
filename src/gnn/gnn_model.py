#fibromyalgia-diagnostic-tool/gan-cnn-project/src/gnn/gnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphSAGE, GATConv, global_mean_pool, global_max_pool

class GNN(nn.Module):
    def __init__(self, model_type='GCN', input_dim=3, hidden_dim=16, output_dim=64, pooling=True):
        super(GNN, self).__init__()
        self.pooling = pooling
        self.dropout = torch.nn.Dropout(p=0.5)  # Dropout layer
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)

        if model_type == 'GCN':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif model_type == 'GraphSAGE':
            self.conv1 = GraphSAGE(input_dim, hidden_dim)
            self.conv2 = GraphSAGE(hidden_dim, hidden_dim)
        elif model_type == 'GAT':
            self.conv1 = GATConv(input_dim, hidden_dim)
            self.conv2 = GATConv(hidden_dim, hidden_dim)
        else:
            raise ValueError("Invalid model type. Choose from 'GCN', 'GraphSAGE', 'GAT'.")
        
        self.fc = nn.Linear(hidden_dim, output_dim)  # Produce 64 features for spatial representation

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch if hasattr(data, 'batch') else None

        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        if self.pooling and batch is not None:
            x = global_mean_pool(x, batch)  # Or use global_max_pool for experimentation
        
        x = self.fc(x)
        return x