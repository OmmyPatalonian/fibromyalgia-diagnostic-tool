from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout

def build_discriminator(data_dim):
    model = Sequential([
        Dense(256, input_dim=data_dim),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(128),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model