'''
Code is built on top of DeepXplore code base.
Density objective and VAE validation is added to the original objective function.
We use DeepXplore as a baseline technique for test generation.

DeepXplore: https://github.com/peikexin9/deepxplore
'''
from __future__ import print_function
import os

from keras.engine.saving import model_from_json

from MNIST.utils import neuron_select
from neuron_select.neural_networks.Model1 import Model1

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定所用的gpu


import argparse

from keras.layers import Input
from keras.utils import to_categorical, np_utils

from utils import *
import imageio
import numpy as np
import datetime
K.clear_session()
batch_size = 256
nb_epoch = 10

# input image dimensions
img_rows, img_cols = 28, 28
img_dim = img_rows * img_cols
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
input_tensor = Input(shape=input_shape)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
model1 = Model1(input_tensor=input_tensor)
print(model1.summary())

# model_name = "LeNet1"
# model_path = os.path.join(os.path.dirname(__file__), 'neural_networks/' + model_name)
#
# json_file = open(model_path + '.json', 'r')  # Read Keras model parameters (stored in JSON file)
# file_content = json_file.read()
# json_file.close()
# model = model_from_json(file_content)  # 从json文件格式中读取文件内容
#
# if not os.path.exists(model_path+'_mnist' + '.hdf5'):
#     model.load_weights(model_path + '.h5')  # 模型加载权重
#     # Compile the model before using 在使用模型前对模型进行编译
#     # model.compile()方法用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#     # trainig
#     model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=nb_epoch, verbose=1)
#     # save model
#     model.save_weights(model_path+'_mnist.hdf5')
#
# else:
#     # 加载整个模型结构
#     model.load_weights(model_path + '_mnist.hdf5')  # 模型加载权重

print("###########model1##################")
for layer in model1.layers:
    model1.summary()
    print(layer.name)
    print(len(layer.get_weights()))

layer_name1, neurons_index1 = neuron_select(model1,"Model1", x_test, y_test)
print(layer_name1)
print(neurons_index1.shape)
