import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBinaryCrossEntropy(nn.Module):
    def __init__(self, pos_weight):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight)
        return loss

class FibroCNN(nn.Module):
    def __init__(self, input_dim, pos_weight):
        super(FibroCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.loss_fn = WeightedBinaryCrossEntropy(pos_weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def compute_loss(self, outputs, targets):
        return self.loss_fn(outputs, targets)