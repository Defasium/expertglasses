#!/usr/bin/python3
'''ShuffleNet Architecture constructor.

This module implements function, which constructs Convolutional neural network for
metric learning based on ShuffleNetV2 (https://arxiv.org/abs/1807.11164).

This code uses the implementation in Keras from repository:
https://github.com/opconty/keras-shufflenetV2

Code have minor changes, which allows to interpolate the bottleneck ratio.

Examples:
    To use this module, you simply import class in your python code:
        # from model_architecture import build_network

    To build a model for images with sizes 64x64x3, use the following code:
        # model = build_network(input_shape=(64, 64, 3), embedding_size=16)

Todo:
    * Add more functionality

.. _Expert eyeglasses recommendation system with Generative Adversarial Networks:
   https://github.com/Defasium/expertglasses

'''

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Dense, Dropout
from shufflenet_and_gans.shufflenetv2 import ShuffleNetV2


def build_network(input_shape, embedding_size):
    '''Api-request to face++ to get various attributes and head orientation.

                Args:
                    input_shape (tuple of int): Input shape of images.
                    embedding_size (int): Size of the final embedding layer.

                Returns:
                    model (tensorflow.keras.engine.training.Model): Keras model.

    '''
    inputs, outputs = ShuffleNetV2(include_top=False, input_shape=input_shape,
                                   bottleneck_ratio=0.35, num_shuffle_units=[2, 2, 2])
    outputs = Dropout(0.0)(outputs)
    outputs = Dense(embedding_size, activation=None,
                    kernel_initializer='he_uniform')(outputs)
    # force the encoding to live on the d-dimentional hypershpere
    outputs = Lambda(lambda x: K.l2_normalize(x, axis=-1))(outputs)
    return Model(inputs, outputs)
