# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:40:17 2019

@author: qq
"""
import os
import sys

from keras import Input
from scipy.misc import imsave
sys.path.append(os.path.join(os.path.dirname(__file__)))
from MNIST.model.Model1 import Model1
from MNIST.model.Model2 import Model2
from MNIST.model.Model3 import Model3
from MNIST.model.Model4 import Model4

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定所用的gpu

import warnings

warnings.filterwarnings("ignore")
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

#当allow_growth设置为True时，分配器将不会指定所有的GPU内存，而是根据需求增长
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import scipy.io as sio
import numpy as np
from keras.datasets import mnist

CLIP_MAX = 0.5
basedir = os.path.abspath(os.path.dirname(__file__))

def deprocess_image(x):
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape(x.shape[1], x.shape[2])  # original shape (1,img_rows, img_cols,1)

def datapredict(dataset='mnist'):
    if dataset == 'mnist':
        # input image dimensions
        img_rows, img_cols = 28, 28
        input_shape = (img_rows, img_cols, 1)

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # define input tensor as a placeholder
        input_tensor = Input(shape=input_shape)

    elif dataset == 'svhn':  # 训练集：73257张图像；测试集：26032张图像。
        train = sio.loadmat("dataset/svhn/svhn_data/svhn_train.mat")
        x_train = np.transpose(train['X'], axes=[3, 0, 1, 2])
        x_train = x_train.astype('float32') / 255.0
        y_train = np.reshape(train['y'], (-1,)) - 1

        test = sio.loadmat("dataset/svhn/svhn_data/svhn_test.mat")
        x_test = np.transpose(test['X'], axes=[3, 0, 1, 2])
        x_test = x_test.astype('float32') / 255.0
        y_test = np.reshape(test['y'], (-1,)) - 1
    return x_train, y_train, x_test, y_test,input_tensor

## deepgini
def select_deepgini_old(dataset, model, selectsize, x_test, y_test):
    if dataset == 'mnist' or dataset == 'fashion_mnist':
        x = np.zeros((selectsize, 28, 28, 1))
        y = np.zeros((selectsize,))
    elif dataset == 'cifar10' or dataset == 'svhn':
        x = np.zeros((selectsize, 32, 32, 3))
        y = np.zeros((selectsize,))

    act_layers = model.predict(x_test)
    metrics = np.sum(act_layers ** 2, axis=1)
    rank_lst = np.argsort(metrics)
    index = 0
    for i in range(len(rank_lst)):
        gen_img = x_test[rank_lst[i]].reshape(-1,28,28,1)
        pred1 = model.predict(gen_img)
        label1 = np.argmax(pred1[0])
        if label1 == y_test[rank_lst[i]] and index < selectsize:
            x[index] = x_test[rank_lst[i]]
            y[index] = y_test[rank_lst[i]]
            index += 1
        if index >= selectsize:
            break

    return x, y, rank_lst

def select_deepgini(dataset, model, selectsize, x_test, y_test):

    x = np.empty(shape=[0, 28, 28, 1])
    y = np.empty(shape=[0,])

    x_test_array = [[] for i in range(10)]
    y_test_array = [[] for i in range(10)]
    class_nums = int(selectsize // 10)

    act_layers = model.predict(x_test)
    metrics = np.sum(act_layers ** 2, axis=1)
    rank_lst = np.argsort(metrics)
    y_ar = y_test[0]
    print(y_test.shape)
    for i in range(len(rank_lst)):
        gen_img = x_test[rank_lst[i]].reshape(-1, 28, 28, 1)
        pred1 = model.predict(gen_img)
        label1 = np.argmax(pred1[0])
        stop = True
        if label1 == y_test[rank_lst[i]] and len(x_test_array[y_test[rank_lst[i]]]) < class_nums:
        # if len(x_test_array[y_test[rank_lst[i]]]) < class_nums:
            x_test_array[y_test[rank_lst[i]]].append(x_test[rank_lst[i]])
            y_test_array[y_test[rank_lst[i]]].append(y_test[rank_lst[i]])

        for j in range(len(x_test_array)):
            if len(x_test_array[j]) != class_nums:
                stop = False
        if stop:
            break

    for i in range(len(x_test_array)):
        x = np.concatenate((x,x_test_array[i]))
        y = np.concatenate((y,y_test_array[i]))

    return x, y, rank_lst

def select_random(dataset, selectsize, x_test, y_test):

    x = np.zeros((selectsize, 28, 28, 1))
    y = np.zeros((selectsize,))


    rank_lst = np.random.choice(range(len(x_test)), selectsize, replace=False)

    for i in range(len(rank_lst)):
        x[i] = x_test[rank_lst[i]]
        y[i] = y_test[rank_lst[i]]

    return x, y, rank_lst

if __name__ == "__main__":
    dataset = 'mnist'
    seed_nums = 50
    model_name = 'Model4'
    x_train, y_train, x_test, y_test, input_tensor = datapredict(dataset)
    if model_name == 'none':
        save_dir = './seeds_50_' + 'random' + '/'
    else:
        save_dir = './seeds_50_' + model_name + '/'
        if model_name == 'Model1':
            model = Model1(input_tensor=input_tensor)
        elif model_name == 'Model2':
            model = Model2(input_tensor=input_tensor)
        elif model_name == 'Model3':
            model = Model3(input_tensor=input_tensor)
        elif model_name == 'Model4':
            model = Model4(input_tensor=input_tensor)

    if os.path.exists(save_dir):
        for i in os.listdir(save_dir):
            path_file = os.path.join(save_dir, i)
            if os.path.isfile(path_file):
                os.remove(path_file)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if model_name == 'none':
        x_select, y_select, rank_lst = select_random('mnist', seed_nums, x_test, y_test)
    else:
        x_select, y_select, rank_lst = select_deepgini('mnist',model,seed_nums,x_test,y_test)
    print(x_select.shape)
    print(y_select.shape)

    for current_sum in range(seed_nums):
        # below logic is to track number of iterations under progress

        gen_img = np.expand_dims(x_select[current_sum], axis=0)
        gen_img_deprocessed = deprocess_image(gen_img)

        seed_index = current_sum
        seed_label = int(y_select[current_sum])
        save_img = save_dir + str(seed_index) + '_' + str(seed_label) + '.png'

        imsave(save_img, gen_img_deprocessed)