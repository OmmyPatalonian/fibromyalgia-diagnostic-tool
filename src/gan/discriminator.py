import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout

def build_discriminator(data_dim, num_classes):
    data_input = tf.keras.Input(shape=(data_dim,))
    label_input = tf.keras.Input(shape=(1,))
    label_embedding = tf.keras.layers.Embedding(num_classes, data_dim)(label_input)
    label_flat = tf.keras.layers.Flatten()(label_embedding)

    merged_input = tf.keras.layers.Concatenate()([data_input, label_flat])
    x = tf.keras.layers.Dense(128, activation="relu")(merged_input)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model([data_input, label_input], x)
