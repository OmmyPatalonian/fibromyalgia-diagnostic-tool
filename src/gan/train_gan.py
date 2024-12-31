import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def train_gan(generator, discriminator, gan, training_data, epochs, batch_size, latent_dim):
    print("Starting GAN training...")
    half_batch = int(batch_size / 2)
    training_history = {'d_loss': [], 'g_loss': []}

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, training_data.shape[0], half_batch)
        real_samples = training_data[idx]
        real_labels = np.ones((half_batch, 1))

        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_labels = np.zeros((half_batch, 1))
        fake_samples = generator.predict([noise, fake_labels])

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Real samples shape: {real_samples.shape}, Real labels shape: {real_labels.shape}")
        print(f"Fake samples shape: {fake_samples.shape}, Fake labels shape: {fake_labels.shape}")

        try:
            d_loss_real = discriminator.train_on_batch([real_samples, real_labels], np.ones((half_batch, 1)))
            d_loss_fake = discriminator.train_on_batch([fake_samples, fake_labels], np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        except Exception as e:
            print(f"Error during discriminator training: {e}")
            print(f"Real samples: {real_samples}")
            print(f"Real labels: {real_labels}")
            print(f"Fake samples: {fake_samples}")
            print(f"Fake labels: {fake_labels}")
            return

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.ones((batch_size, 1))
        fake_labels = np.random.randint(0, 2, (batch_size, 1))

        try:
            g_loss = gan.train_on_batch([noise, fake_labels], valid_y)
        except Exception as e:
            print(f"Error during generator training: {e}")
            print(f"Noise: {noise}")
            print(f"Valid labels: {valid_y}")
            return

        training_history['d_loss'].append(d_loss[0])
        training_history['g_loss'].append(g_loss)

        print(f"{epoch+1} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")

        # Visualize generated samples
        if (epoch + 1) % 100 == 0:
            generated_samples = generator.predict([np.random.normal(0, 1, (100, latent_dim)), np.random.randint(0, 2, (100, 1))])
            plt.scatter(generated_samples[:, 0], generated_samples[:, 1], label="Generated Data")
            plt.legend()
            plt.show()

    print("GAN training completed.")

    # Plot training history
    plt.plot(training_history['d_loss'], label='Discriminator Loss')
    plt.plot(training_history['g_loss'], label='Generator Loss')
    plt.legend()
    plt.show()