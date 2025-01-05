import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, data_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

def build_generator(latent_dim, data_dim):
    return Generator(latent_dim, data_dim)