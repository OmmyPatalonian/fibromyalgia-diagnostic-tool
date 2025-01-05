import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_channels=1, output_dim=1, dropout_rate=0.3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 500, output_dim)  # Adjust input size based on your data
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.unsqueeze(1)  # Assumes input is 1D; ensure this matches your data
        x = self.pool(F.leaky_relu(self.conv1(x), negative_slope=0.1))
        x = x.view(-1, 16 * 500)  # Adjust input size based on your data
        x = self.dropout(x)
        x = torch.sigmoid(self.fc1(x))
        return x

class CNN2D(nn.Module):
    def __init__(self, input_channels=3, output_dim=1, dropout_rate=0.3):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, output_dim)  # Adjust based on the output size after pooling
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = self.pool(x)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(x)
        x = torch.sigmoid(self.fc1(x))  # Binary classification
        return x