from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU

def build_autoencoder(input_dim):
    print("Building autoencoder...")
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128)(input_layer)
    encoded = LeakyReLU(alpha=0.1)(encoded)
    encoded = Dense(64)(encoded)
    encoded = LeakyReLU(alpha=0.1)(encoded)
    encoded = Dense(32)(encoded)
    encoded = LeakyReLU(alpha=0.1)(encoded)
    decoded = Dense(64)(encoded)
    decoded = LeakyReLU(alpha=0.1)(decoded)
    decoded = Dense(128)(decoded)
    decoded = LeakyReLU(alpha=0.1)(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    print("Autoencoder built successfully.")
    return Model(input_layer, decoded)