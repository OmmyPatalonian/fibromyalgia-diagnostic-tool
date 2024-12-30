import numpy as np

def train_gan(generator, discriminator, gan, data, epochs, batch_size, latent_dim):
    for epoch in range(epochs):
        # Step 1: Train Discriminator
        real_samples = data[np.random.randint(0, data.shape[0], batch_size)]
        fake_samples = generator.predict(np.random.normal(0, 1, (batch_size, latent_dim)))
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)

        # Step 2: Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))  # Train against "real" label

        # Log progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, D Loss: {(d_loss_real + d_loss_fake) / 2}, G Loss: {g_loss}")