import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout
def Alexnet8(classes):
    model = tf.keras.Sequential([
        # Conv2D(filters=96, kernel_size=(3,3), activation='relu', input_shape=(30, 105, 1)),
        Conv2D(filters=96, kernel_size=(3,3), activation='relu'),
        BatchNormalization(),
        MaxPool2D(pool_size=(3,3), strides=2, padding='same'),

        Conv2D(256, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPool2D((3,3), 2, padding='same'),

        Conv2D(384, (3,3), padding='same', activation='relu'),

        Conv2D(384, (3, 3), padding='same', activation='relu'),

        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPool2D((3,3), 2, padding='same'),

        Flatten(),
        Dense(2048, activation='relu'),
        Dropout(0.5),
        Dense(2048, activation='relu'),
        Dropout(0.5),
        Dense(classes, activation='softmax')
    ])
    return model