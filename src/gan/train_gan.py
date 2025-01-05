import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)

def gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = tf.random.uniform([real_samples.shape[0], 1], 0.0, 1.0)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        validity_interpolated = discriminator(interpolated)
    gradients = tape.gradient(validity_interpolated, [interpolated])[0]
    gradients_sqr = tf.square(gradients)
    gradient_penalty = tf.reduce_mean(gradients_sqr)
    return gradient_penalty

def train_gan(generator, discriminator, real_data, epochs=100, batch_size=64, latent_dim=100):
    # Convert real data to tensor
    real_data = torch.tensor(real_data, dtype=torch.float32)
    real_labels = torch.ones((real_data.size(0), 1), dtype=torch.float32)
    fake_labels = torch.zeros((real_data.size(0), 1), dtype=torch.float32)

    # Create DataLoader for real data
    dataset = TensorDataset(real_data, real_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizers
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for i, (real_samples, _) in enumerate(dataloader):
            batch_size = real_samples.size(0)

            # Train discriminator
            optimizer_d.zero_grad()

            # Real samples
            real_samples = real_samples.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            real_labels = torch.ones((batch_size, 1), dtype=torch.float32).to(real_samples.device)
            output_real = discriminator(real_samples)
            loss_real = criterion(output_real, real_labels)

            # Fake samples
            noise = torch.randn((batch_size, latent_dim)).to(real_samples.device)
            fake_samples = generator(noise)
            fake_labels = torch.zeros((batch_size, 1), dtype=torch.float32).to(real_samples.device)
            output_fake = discriminator(fake_samples.detach())
            loss_fake = criterion(output_fake, fake_labels)

            # Total discriminator loss
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            # Train generator
            optimizer_g.zero_grad()

            # Generate fake samples
            output_fake = discriminator(fake_samples)
            loss_g = criterion(output_fake, real_labels)  # We want the generator to fool the discriminator
            loss_g.backward()
            optimizer_g.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], "
                      f"D Loss: {loss_d.item():.4f}, G Loss: {loss_g.item():.4f}")

    print("GAN training completed.")

# Example usage
if __name__ == "__main__":
    # Define your generator, discriminator, and gan models here
    # Ensure the discriminator has a sigmoid activation in the last layer
    # Ensure the generator's final activation is tanh and the real data is normalized to [-1, 1]
    pass