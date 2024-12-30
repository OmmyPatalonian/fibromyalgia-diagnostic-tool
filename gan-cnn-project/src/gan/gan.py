from tensorflow.keras.models import Sequential

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential([generator, discriminator])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                  loss='binary_crossentropy')
    return model