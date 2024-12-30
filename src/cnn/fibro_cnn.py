import torch

class FibroCNN(torch.nn.Module):
    def __init__(self, input_dim):
        super(FibroCNN, self).__init__()
        print("Initializing FibroCNN...")
        self.conv1 = torch.nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(32 * input_dim, 128)
        self.fc2 = torch.nn.Linear(128, input_dim)
        print("FibroCNN initialized.")
        
    def forward(self, x):
        print("Forward pass through FibroCNN...")
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        output = self.fc2(x)
        print("Forward pass completed.")
        return output