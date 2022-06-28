from tensorflow.keras.layers import *
import tensorflow as tf

def ResnetBlock(inputs, filters, strides=1, residual_path=False):
    residual = inputs
    x = Conv2D(filters, (3,3), strides=strides, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3,3), strides=1, padding='same', use_bias=False)(x)
    y = BatchNormalization()(x)

    if residual_path:
        residual = Conv2D(filters, (1,1), strides=strides, padding='same', use_bias=False)(inputs)
        residual = BatchNormalization()(residual)

    out = Activation('relu')(y+residual)
    return out

def ResNet18(frame, nclass, shape_y):
    inputs = tf.keras.Input(shape=(frame, shape_y, 1))
    x = Conv2D(64, (3,3), strides=1, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    block = ResnetBlock(x, 64, residual_path=False)
    block = ResnetBlock(block, 64, residual_path=False)

    block = ResnetBlock(block, 128, 2, residual_path=True)
    block = ResnetBlock(block, 128, residual_path=False)

    block = ResnetBlock(block, 256, 2, residual_path=True)
    block = ResnetBlock(block, 256, residual_path=False)

    block = ResnetBlock(block, 512, 2, residual_path=True)
    block = ResnetBlock(block, 512, residual_path=False)

    x = GlobalAveragePooling2D()(block)
    y = Dense(nclass, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())(x)
    model = tf.keras.Model(inputs=inputs, outputs=y)
    return model
# model = ResNet18(30, 9, 45)
# model.summary()