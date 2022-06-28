import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Concatenate, MaxPooling2D, Lambda, Input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
import tensorflow.keras.models as models
import tensorflow.keras.backend as backend

def conv2d_cell(x, nbfilter, filtersz, strides=1, pad='same', act=True, name=None):
    x = Conv2D(nbfilter, filtersz, strides, padding=pad, use_bias=False, data_format='channels_last', name=name+'conv2d')(x)
    x = BatchNormalization(axis=3, scale=False, name=name+'conv2dbn')(x)

    if act:
        x = Activation('relu', name=name+'conv2dact')(x)
    return x


def stem_block(x, name=None):
    x = conv2d_cell(x, 32, 3, 2, 'same', True, name=name + 'conv1')
    x = conv2d_cell(x, 32, 3, 1, 'same', True, name=name + 'conv2')
    x = conv2d_cell(x, 64, 3, 1, 'same', True, name=name + 'conv3')

    x_11 = MaxPooling2D(3, strides=2, padding='same', name=name + '_branch11_maxpool')(x)
    x_12 = conv2d_cell(x, 64, 3, 2, 'same', True, name=name + '_branch12')

    x = Concatenate(axis=3, name=name + 'concat_1')([x_11, x_12])

    x_21 = conv2d_cell(x, 64, 1, 1, 'same', True, name=name + '_branch211')
    x_21 = conv2d_cell(x_21, 64, [1, 7], 1, 'same', True, name=name + '_branch212')
    x_21 = conv2d_cell(x_21, 64, [7, 1], 1, 'same', True, name=name + '_branch213')
    x_21 = conv2d_cell(x_21, 96, 3, 1, 'same', True, name=name + '_branch214')

    x_22 = conv2d_cell(x, 64, 1, 1, 'same', True, name=name + '_branch221')
    x_22 = conv2d_cell(x_22, 96, 3, 1, 'same', True, name=name + '_brach222')

    x = Concatenate(axis=3, name=name + 'stem_concat_2')([x_21, x_22])

    x_31 = conv2d_cell(x, 192, 3, 2, 'same', True, name=name + '_brach31')
    x_32 = MaxPooling2D(3, strides=2, padding='same', name=name + '_branch32_maxpool')(x)


    x = Concatenate(axis=3, name=name + 'stem_concat_3')([x_31, x_32])

    return x


def incresA(x, scale, name=None):
    branch0 = conv2d_cell(x, 32, 1, 1, 'same', True, name=name + 'b0')

    branch1 = conv2d_cell(x, 32, 1, 1, 'same', True, name=name + 'b1_1')
    branch1 = conv2d_cell(branch1, 32, 3, 1, 'same', True, name=name + 'b1_2')

    branch2 = conv2d_cell(x, 32, 1, 1, 'same', True, name=name + 'b2_1')
    branch2 = conv2d_cell(branch2, 32, 3, 1, 'same', True, name=name + 'b2_2')
    branch2 = conv2d_cell(branch2, 32, 3, 1, 'same', True, name=name + 'b2_3')

    branches = [branch0, branch1, branch2]
    mixed = Concatenate(axis=3, name=name + '_concat')(branches)
    filt_exp_1x1 = conv2d_cell(mixed, 384, 1, 1, 'same', False, name=name + 'filt_exp_1x1')

    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                       output_shape=backend.int_shape(x)[1:],
                       arguments={'scale': scale},
                       name=name + 'act_scaling')(
        [x, filt_exp_1x1])  # Lambda https://blog.csdn.net/VeritasCN/article/details/89884036

    return final_lay

def incresB(x, scale, name=None):

    branch0 = conv2d_cell(x, 192, 1, 1, 'same', True, name= name+'b0')

    branch1 = conv2d_cell(x, 128, 1, 1, 'same', True, name=name+'b1_1')
    branch1 = conv2d_cell(branch1, 160, [1,7], 1, 'same', True, name=name+'b1_2')
    branch1 = conv2d_cell(branch1, 192, [7,1], 1, 'same', True, name=name+'b1_3')

    branches = [branch0, branch1]
    mixed = Concatenate(axis=3, name=name+'_mixed')(branches)
    filt_exp_1x1 = conv2d_cell(mixed, 1152, 1 ,1, 'same', False, name=name+'filt_exp_1x1')
    final_lay = Lambda(lambda inputs, scale:inputs[0]+inputs[1]*scale,
                        output_shape=backend.int_shape(x)[1:],
                        arguments = {'scale':scale},
                        name = name+'act_scaling')([x, filt_exp_1x1])

    return final_lay


def incresC(x, scale, name=None):

    branch0 = conv2d_cell(x, 192, 1, 1, 'same', True, name= name+'b0')

    branch1 = conv2d_cell(x, 192, 1, 1, 'same', True, name=name+'b1_1')
    branch1 = conv2d_cell(branch1, 224, [1,3], 1, 'same', True, name=name+'b1_2')
    branch1 = conv2d_cell(branch1, 256, [3,1], 1, 'same', True, name=name+'b1_3')

    branches = [branch0, branch1]
    mixed = Concatenate(axis=3, name=name+'_mixed')(branches)
    filt_exp_1x1 = conv2d_cell(mixed, 2048, 1 ,1, 'same', False, name=name+'filt_exp_1x1')
    final_lay = Lambda(lambda inputs, scale:inputs[0]+inputs[1]*scale,
                        output_shape=backend.int_shape(x)[1:],
                        arguments = {'scale':scale},
                        name = name+'act_scaling')([x, filt_exp_1x1])

    return final_lay


def reductionA (x, name=None):

    x_1 = MaxPooling2D(2, strides=2, padding='same', name=name+'branch1')(x)

    x_2 = conv2d_cell(x, 384, 3, 2, 'same', True, name=name+'branch2')

    x_3 = conv2d_cell(x, 256, 1, 1, 'same', True, name=name+'branch31')
    x_3 = conv2d_cell(x_3, 256, 3, 1, 'same', True, name=name+'branch32')
    x_3 = conv2d_cell(x_3, 384, 3, 2, 'same', True, name=name+'branch33')

    x = Concatenate(axis=3, name=name+'concat')([x_1, x_2, x_3])

    return x


def reductionB(x, name=None):

    x_1 = MaxPooling2D(3, strides=2, padding='same', name=name+'branch1')(x)

    x_2 = conv2d_cell(x, 256, 1, 1, 'same', True, name=name+'branch2')
    x_2 = conv2d_cell(x, 384, 3, 2, 'same', True, name=name+'branch2')

    x_3 = conv2d_cell(x, 256, 1, 1, 'same', True, name=name+'branch31')
    x_3 = conv2d_cell(x_3, 256, 3, 2, 'same', True, name=name+'branch32')

    x_4 = conv2d_cell(x, 256, 1, 1, 'same', True, name=name+'branch41')
    x_4 = conv2d_cell(x_4, 256, 3, 1, 'same', True, name=name+'branch42')
    x_4 = conv2d_cell(x_4, 256, 3, 2, 'same', True, name=name+'branch43')

    x = Concatenate(axis=3, name=name+'concat')([x_1, x_2, x_3, x_4])

    return x


def InceptionResNet_V2(img_shape, num_classes):
    img_input = Input(shape=img_shape)
    x = stem_block(img_input, name='stem')

    # Inception-ResNet-A modules
    x = incresA(x, 0.15, name='incresA_1')
    x = incresA(x, 0.15, name='incresA_2')
    x = incresA(x, 0.15, name='incresA_3')
    x = incresA(x, 0.15, name='incresA_4')
    x = incresA(x, 0.15, name='incresA_5')

    # Inception-ResNet-ReductionA modules
    x = reductionA(x, name='reductionA')

    # Inception-ResNet-B modules
    x = incresB(x, 0.1, name='incresB_1')
    x = incresB(x, 0.1, name='incresB_2')
    x = incresB(x, 0.1, name='incresB_3')
    x = incresB(x, 0.1, name='incresB_4')
    x = incresB(x, 0.1, name='incresB_5')
    x = incresB(x, 0.1, name='incresB_6')
    x = incresB(x, 0.1, name='incresB_7')
    x = incresB(x, 0.1, name='incresB_8')
    x = incresB(x, 0.1, name='incresB_9')
    x = incresB(x, 0.1, name='incresB_10')

    # Inception-ResNet-ReductionB modules
    x = reductionB(x, name='reductionB')

    # Inception-ResNet-C modules
    x = incresC(x, 0.2, name='incresC_1')
    x = incresC(x, 0.2, name='incresC_2')
    x = incresC(x, 0.2, name='incresC_3')
    x = incresC(x, 0.2, name='incresC_4')
    x = incresC(x, 0.2, name='incresC_5')

    # Top layer
    x = GlobalAveragePooling2D(data_format='channels_last')(x)
    x = Dropout(0.6)(x)
    output = Dense(num_classes, activation='softmax')(x)

    # build Model
    model = models.Model(img_input, output, name='InceptionResNet_V2')

    return model


# test main()
# def main():
#     model = InceptionResNet_V2((299, 299, 3), 1000)
#     model.summary()
# model = InceptionResNet_V2((30, 147, 1), 9)
# model.summary()

# if __name__ == '__main__':
#     main()