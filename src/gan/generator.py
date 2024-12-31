import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout

def build_generator(latent_dim, data_dim, num_classes):
    label_input = tf.keras.Input(shape=(1,))
    label_embedding = tf.keras.layers.Embedding(num_classes, latent_dim)(label_input)
    label_flat = tf.keras.layers.Flatten()(label_embedding)

    noise_input = tf.keras.Input(shape=(latent_dim,))
    merged_input = tf.keras.layers.Concatenate()([noise_input, label_flat])
    x = tf.keras.layers.Dense(128, activation="relu")(merged_input)
    x = tf.keras.layers.Dense(data_dim, activation="tanh")(x)
    return tf.keras.Model([noise_input, label_input], x)