import os
import tensorflow as tf

from pathlib import Path
from functools import partial

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import add
from tensorflow.keras import backend as K


def scaling(x, scale):
    return x * scale


def stem(inputs):
    x = Conv2D(32, 3, strides=2, padding='valid', use_bias=False)(inputs)
    x = BatchNormalization(axis=3, momentum=0.995,
                           epsilon=0.001, scale=False)(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, strides=1, padding='valid', use_bias=False)(x)
    x = BatchNormalization(axis=3, momentum=0.995,
                           epsilon=0.001, scale=False)(x)
    x = Activation('relu')(x)
    x = Conv2D(64, 3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization(axis=3, momentum=0.995,
                           epsilon=0.001, scale=False)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(3, strides=2)(x)
    x = Conv2D(80, 1, strides=1, padding='valid', use_bias=False)(x)
    x = BatchNormalization(axis=3, momentum=0.995,
                           epsilon=0.001, scale=False)(x)
    x = Activation('relu')(x)
    x = Conv2D(192, 3, strides=1, padding='valid', use_bias=False)(x)
    x = BatchNormalization(axis=3, momentum=0.995,
                           epsilon=0.001, scale=False)(x)
    x = Activation('relu')(x)
    x = Conv2D(256, 3, strides=2, padding='valid', use_bias=False)(x)
    x = BatchNormalization(axis=3, momentum=0.995,
                           epsilon=0.001, scale=False)(x)
    x = Activation('relu')(x)

    return x


def incetption_resnet_A(x):
    # Inception-ResNet-A block:
    # Branch 0
    branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False)(x)
    branch_0 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_0)
    branch_0 = Activation('relu')(branch_0)

    # Branch 1
    branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False)(x)
    branch_1 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_1)
    branch_1 = Activation('relu')(branch_1)
    branch_1 = Conv2D(32, 3, strides=1, padding='same',
                      use_bias=False)(branch_1)
    branch_1 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_1)
    branch_1 = Activation('relu')(branch_1)

    # Branch 2
    branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False)(x)
    branch_2 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_2)
    branch_2 = Activation('relu')(branch_2)
    branch_2 = Conv2D(32, 3, strides=1, padding='same',
                      use_bias=False)(branch_2)
    branch_2 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_2)
    branch_2 = Activation('relu')(branch_2)
    branch_2 = Conv2D(32, 3, strides=1, padding='same',
                      use_bias=False)(branch_2)
    branch_2 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_2)
    branch_2 = Activation('relu')(branch_2)
    branches = [branch_0, branch_1, branch_2]
    mixed = Concatenate(axis=3)(branches)
    up = Conv2D(256, 1, strides=1, padding='same', use_bias=True)(mixed)
    up = Lambda(scaling, output_shape=K.int_shape(up)
                [1:], arguments={'scale': 0.17})(up)
    x = add([x, up])
    x = Activation('relu')(x)

    return x


def reduction_block_A(x):
    # Mixed 6a (Reduction-A block):
    # Branch 0
    branch_0 = Conv2D(384, 3, strides=2, padding='valid', use_bias=False)(x)
    branch_0 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_0)
    branch_0 = Activation('relu')(branch_0)

    # Branch 1
    branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False)(x)
    branch_1 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_1)
    branch_1 = Activation('relu')(branch_1)
    branch_1 = Conv2D(192, 3, strides=1, padding='same',
                      use_bias=False)(branch_1)
    branch_1 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_1)
    branch_1 = Activation('relu')(branch_1)
    branch_1 = Conv2D(256, 3, strides=2, padding='valid',
                      use_bias=False)(branch_1)
    branch_1 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_1)
    branch_1 = Activation('relu')(branch_1)
    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=3)(branches)

    return x


def incetption_resnet_B(x):
    # (Inception-ResNet-B block):
    # Branch 0
    branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False)(x)
    branch_0 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_0)
    branch_0 = Activation('relu')(branch_0)

    # Branch 1
    branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False)(x)
    branch_1 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_1)
    branch_1 = Activation('relu')(branch_1)
    branch_1 = Conv2D(128, [1, 7], strides=1,
                      padding='same', use_bias=False)(branch_1)
    branch_1 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_1)
    branch_1 = Activation('relu')(branch_1)
    branch_1 = Conv2D(128, [7, 1], strides=1,
                      padding='same', use_bias=False)(branch_1)
    branch_1 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_1)
    branch_1 = Activation('relu')(branch_1)
    branches = [branch_0, branch_1]
    mixed = Concatenate(axis=3)(branches)
    up = Conv2D(896, 1, strides=1, padding='same', use_bias=True)(mixed)
    up = Lambda(scaling, output_shape=K.int_shape(up)
                [1:], arguments={'scale': 0.1})(up)
    x = add([x, up])
    x = Activation('relu')(x)

    return x


def reduction_block_B(x):
    # Branch 0
    branch_0 = Conv2D(256, 1, strides=1, padding='same', use_bias=False)(x)
    branch_0 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_0)
    branch_0 = Activation('relu')(branch_0)
    branch_0 = Conv2D(384, 3, strides=2, padding='valid',
                      use_bias=False)(branch_0)
    branch_0 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_0)
    branch_0 = Activation('relu')(branch_0)

    # Branch 1
    branch_1 = Conv2D(256, 1, strides=1, padding='same', use_bias=False)(x)
    branch_1 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_1)
    branch_1 = Activation('relu')(branch_1)
    branch_1 = Conv2D(256, 3, strides=2, padding='valid',
                      use_bias=False)(branch_1)
    branch_1 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_1)
    branch_1 = Activation('relu')(branch_1)

    # Branch 2
    branch_2 = Conv2D(256, 1, strides=1, padding='same', use_bias=False)(x)
    branch_2 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_2)
    branch_2 = Activation('relu')(branch_2)
    branch_2 = Conv2D(256, 3, strides=1, padding='same',
                      use_bias=False)(branch_2)
    branch_2 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_2)
    branch_2 = Activation('relu')(branch_2)
    branch_2 = Conv2D(256, 3, strides=2, padding='valid',
                      use_bias=False)(branch_2)
    branch_2 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_2)
    branch_2 = Activation('relu')(branch_2)
    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=3)(branches)

    return x


def incetption_resnet_C(x):
    # Branch 0
    branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False)(x)
    branch_0 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_0)
    branch_0 = Activation('relu')(branch_0)

    # Branch 1
    branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False)(x)
    branch_1 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_1)
    branch_1 = Activation('relu')(branch_1)
    branch_1 = Conv2D(192, [1, 3], strides=1,
                      padding='same', use_bias=False)(branch_1)
    branch_1 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_1)
    branch_1 = Activation('relu')(branch_1)
    branch_1 = Conv2D(192, [3, 1], strides=1,
                      padding='same', use_bias=False)(branch_1)
    branch_1 = BatchNormalization(
        axis=3, momentum=0.995, epsilon=0.001, scale=False)(branch_1)
    branch_1 = Activation('relu')(branch_1)
    branches = [branch_0, branch_1]
    mixed = Concatenate(axis=3)(branches)
    up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True)(mixed)
    up = Lambda(scaling, output_shape=K.int_shape(up)
                [1:], arguments={'scale': 0.2})(up)
    x = add([x, up])
    x = Activation('relu')(x)

    return x


def InceptionResNetV1(input_shape):
    inputs = Input(shape=input_shape)

    x = stem(inputs)

    for _ in range(5):
        x = incetption_resnet_A(x)

    x = reduction_block_A(x)

    for _ in range(10):
        x = incetption_resnet_B(x)

    x = reduction_block_B(x)

    for _ in range(5):
        x = incetption_resnet_C(x)

    # Classification block
    x = GlobalAveragePooling2D(name='AvgPool')(x)
    x = Dropout(1.0 - 0.8, name='Dropout')(x)
    # Bottleneck
    x = Dense(128, use_bias=False, name='Bottleneck')(x)
    x = BatchNormalization(momentum=0.995, epsilon=0.001,
                           scale=False, name='Bottleneck_BatchNorm')(x)

    # Create model
    model = Model(inputs, x, name='inception_resnet_v1')

    return model


model = InceptionResNetV1((160, 160, 3))
model.summary()
