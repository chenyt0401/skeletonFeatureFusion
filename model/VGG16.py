import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout, Activation

def VGG16Net(classes):
    model = tf.keras.Sequential([
        Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),  # BN层1
        Activation('relu'),  # 激活层1
        Conv2D(filters=64, kernel_size=(3, 3), padding='same', ),
        BatchNormalization(),  # BN层1
        Activation('relu'),  # 激活层1
        MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
        Dropout(0.2),  # dropout层

        Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),  # BN层1
        Activation('relu'),  # 激活层1
        Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),  # BN层1
        Activation('relu'), # 激活层1
        MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
        Dropout(0.2),  # dropout层

        Conv2D(filters=256, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),  # BN层1
        Activation('relu'),  # 激活层1
        Conv2D(filters=256, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),  # BN层1
        Activation('relu'),  # 激活层1
        Conv2D(filters=256, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
        Dropout(0.2),

        Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),  # BN层1
        Activation('relu'),  # 激活层1
        Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),  # BN层1
        Activation('relu'),  # 激活层1
        Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
        Dropout(0.2),

        Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),  # BN层1
        Activation('relu'),  # 激活层1
        Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),  # BN层1
        Activation('relu'),  # 激活层1
        Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
        Dropout(0.2),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(classes, activation='softmax')
    ])
    return model