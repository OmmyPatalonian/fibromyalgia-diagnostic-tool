import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FibroCNN(nn.Module):
    def __init__(self, input_dim, pos_weight):
        super(FibroCNN, self).__init__()
        # Load pretrained ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Replace the first layer to accept `input_dim` channels
        self.resnet.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the output layer for binary classification
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        
        self.pos_weight = pos_weight

    def forward(self, x):
        # Pass input through ResNet
        logits = self.resnet(x)  # Outputs logits
        return logits

    def compute_loss(self, output, target):
        # Use BCEWithLogitsLoss with positional weight
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        return criterion(output, target)