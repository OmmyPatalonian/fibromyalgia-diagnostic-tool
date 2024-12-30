import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout

def build_discriminator(data_dim):
    print("Building discriminator...")
    model = Sequential([
        Input(shape=(data_dim,)),
        Dense(256),
        LeakyReLU(negative_slope=0.2),
        Dropout(0.3),
        Dense(128),
        LeakyReLU(negative_slope=0.2),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print("Discriminator built and compiled successfully.")
    return model
