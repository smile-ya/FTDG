# usage: python driving_models.py 1 - train the dave-orig model

from __future__ import print_function

import math

from keras.layers import Convolution2D, Input, Dense, Flatten, Lambda, MaxPooling2D, Dropout
import os,sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
print(sys.path)
print(os.getcwd())
print(os.path.dirname(os.path.dirname(__file__)))
print(os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'neuron_select/lrp.py')))
from configs import bcolors
from data_utils import load_train_data, load_test_data
from utils import *

tf.keras.backend.set_learning_phase(True)

def Dave_orig(input_tensor=None, load_weights=False):  # original dave
    if input_tensor is None:
        input_tensor = Input(shape=(100, 100, 3))
    x = Convolution2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2), name='block1_conv1')(input_tensor)
    x = Convolution2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2), name='block1_conv2')(x)
    x = Convolution2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2), name='block1_conv3')(x)
    x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1), name='block1_conv4')(x)
    x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1), name='block1_conv5')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(1164, activation='relu', name='fc1')(x)
    x = Dense(100, activation='relu', name='fc2')(x)
    x = Dense(50, activation='relu', name='fc3')(x)
    x = Dense(10, activation='relu', name='fc4')(x)
    x = Dense(1, name='before_prediction')(x)
    x = Lambda(atan_layer, output_shape=atan_layer_shape, name='prediction')(x)

    model = Model(input_tensor, x)
    if load_weights:
        model.load_weights(os.path.join(os.path.dirname(__file__), 'Model1.h5'))


    # compiling
    model.compile(loss='mse', optimizer='adadelta', metrics=['mae'])
    print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
    return model


def Dave_norminit(input_tensor=None, load_weights=False):  # original dave with normal initialization
    if input_tensor is None:
        input_tensor = Input(shape=(100, 100, 3))
    x = Convolution2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2),
                      name='block1_conv1')(input_tensor)
    x = Convolution2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2),
                      name='block1_conv2')(x)
    x = Convolution2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2),
                      name='block1_conv3')(x)
    x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1),
                      name='block1_conv4')(x)
    x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1),
                      name='block1_conv5')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(1164, kernel_initializer=normal_init, activation='relu', name='fc1')(x)
    x = Dense(100, kernel_initializer=normal_init, activation='relu', name='fc2')(x)
    x = Dense(50, kernel_initializer=normal_init, activation='relu', name='fc3')(x)
    x = Dense(10, kernel_initializer=normal_init, activation='relu', name='fc4')(x)
    x = Dense(1, name='before_prediction')(x)
    x = Lambda(atan_layer, output_shape=atan_layer_shape, name='prediction')(x)

    model = Model(input_tensor, x)
    if load_weights:
        model.load_weights(os.path.join(os.path.dirname(__file__), 'Model2.h5'))

    # compiling
    model.compile(loss='mse', optimizer='adadelta',  metrics=['mae'])
    print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
    return model


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
    x = Dropout(.5)(x, training=True)
    x = Dense(100, activation='relu', name='fc2')(x)
    x = Dropout(.25)(x, training=True)
    x = Dense(20, activation='relu', name='fc3')(x)
    x = Dense(1, name='before_prediction')(x)
    x = Lambda(atan_layer, output_shape=atan_layer_shape, name="prediction")(x)

    model = Model(input_tensor, x)
    if load_weights:
        model.load_weights(os.path.join(os.path.dirname(__file__), 'Model3.h5'))

    # compiling
    model.compile(loss='mse', optimizer='adadelta',  metrics=['mae'])
    print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
    return model


if __name__ == '__main__':
    # train the model
    batch_size = 256
    nb_epoch = 15
    model_name = "1"

    if model_name == '1':
        model = Dave_orig()
        save_model_name = 'Model1.h5'
    elif model_name == '2':
        # K.set_learning_phase(1)
        model = Dave_norminit()
        save_model_name = 'Model2.h5'
    elif model_name == '3':
        # K.set_learning_phase(1)
        model = Dave_dropout()
        save_model_name = 'Model3.h5'
    else:
        print(bcolors.FAIL + 'invalid model name, must one of 1, 2 or 3' + bcolors.ENDC)

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
    model.evaluate_generator(test_generator,
                             steps=math.ceil(samples_per_epoch * 1. / batch_size))
    # save model
    model.save_weights(os.path.join(os.path.dirname(__file__), save_model_name))

