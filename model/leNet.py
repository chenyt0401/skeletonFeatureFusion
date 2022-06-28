from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import tensorflow as tf


def LeNet(classes):  # 改变写法
    inputs = tf.keras.Input(shape=(30, 147, 1))
    x = Conv2D(6, (5, 5), activation='relu', padding='same')(inputs)
    x = MaxPool2D(2,2)(x)
    x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = MaxPool2D(2,2)(x)
    x = Flatten()(x)
    x = Dense(120, activation='relu')(x)
    x = Dense(84, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model