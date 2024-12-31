import tensorflow as tf
from tensorflow.keras.models import Sequential

def build_gan(generator, discriminator):
    print("Building GAN...")
    discriminator.trainable = False
    model = Sequential([generator, discriminator])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                  loss='binary_crossentropy')
    print("GAN built and compiled successfully.")
    return model