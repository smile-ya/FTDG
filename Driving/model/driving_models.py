# usage: python driving_models.py 1 - train the dave-orig model

from __future__ import print_function

import math
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from keras.layers import Convolution2D, Input, Dense, Flatten, Lambda, MaxPooling2D, Dropout
import tensorflow as tf

from configs import bcolors
from data_utils import load_train_data, load_test_data
from utils_tmp import *
tf.keras.backend.set_learning_phase(True) #由于加入dropout层会使得训练和测试时的行为不同，
                                          # 因此,加上这句话可以保证dropout层在训练时和测试时都用上。

def Dave_dropout(input_tensor=None, load_weights=False):  # simplified dave
    if input_tensor is None:
        input_tensor = Input(shape=(100, 100, 3))
    x = Convolution2D(16, (3, 3), padding='valid', activation='relu', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
    x = Convolution2D(32, (3, 3), padding='valid', activation='relu', name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool2')(x)
    x = Convolution2D(64, (3, 3), padding='valid', activation='relu', name='block1_conv3')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool3')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(500, activation='relu', name='fc1')(x)
    x = Dropout(.5,name='drop1')(x, training=True)
    x = Dense(100, activation='relu', name='fc2')(x)
    x = Dropout(.25,name='drop2')(x,training=True)
    x = Dense(20, activation='relu', name='fc3')(x)
    x = Dense(1, name='before_prediction')(x)
    x = Lambda(atan_layer, output_shape=atan_layer_shape, name="prediction")(x)

    m = Model(input_tensor, x)
    if load_weights:
        m.load_weights(os.path.join(os.path.dirname(__file__), 'Dave_dropout.h5'))

    # compiling
    m.compile(loss='mse', optimizer='adadelta', metrics=['mse'])
    print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
    return m


if __name__ == '__main__':
    # train the model
    batch_size = 256
    nb_epoch = 15

    model = Dave_dropout()
    save_model_name = os.path.join(os.path.dirname(__file__), 'Dave_dropout.h5')

    # the data, shuffled and split between train and test sets
    train_generator, samples_per_epoch = load_train_data(batch_size=batch_size, shape=(100, 100))

    # trainig
    model.fit_generator(train_generator,
                        steps_per_epoch=math.ceil(samples_per_epoch * 1. / batch_size),
                        epochs=nb_epoch,
                        workers=8,
                        use_multiprocessing=True)
    print(bcolors.OKGREEN + 'Model trained' + bcolors.ENDC)

    # evaluation
    K.set_learning_phase(0)
    test_generator, samples_per_epoch, _, _ = load_test_data(batch_size=batch_size, shape=(100, 100))
    score = model.evaluate_generator(test_generator,
                             steps=math.ceil(samples_per_epoch * 1. / batch_size))
    print(score) #返回MSE值
    # save model
    model.save_weights(save_model_name)
