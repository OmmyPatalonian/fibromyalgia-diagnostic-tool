from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def build_autoencoder(input_dim):
    print("Building autoencoder...")
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    print("Autoencoder built successfully.")
    return Model(input_layer, decoded)