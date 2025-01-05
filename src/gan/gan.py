import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def build_gan(generator, discriminator):
    discriminator.trainable = False
    noise = Input(shape=(generator.input_shape[0][1],))
    label = Input(shape=(1,), dtype='int32')
    generated_data = generator([noise, label])
    validity = discriminator([generated_data, label])
    model = Model([noise, label], validity)
    model.compile(optimizer=Adam(0.0002, 0.5), loss=wasserstein_loss)
    return model