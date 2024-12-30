from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout

def build_generator(latent_dim, data_dim):
    print("Building generator...")
    model = Sequential([
        Dense(128, input_dim=latent_dim),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(data_dim, activation='tanh')  # Output scaled to [-1, 1]
    ])
    print("Generator built successfully.")
    return model