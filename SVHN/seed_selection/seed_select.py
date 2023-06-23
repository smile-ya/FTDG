# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:40:17 2019

@author: qq
"""
import os
import sys

from cleverhans.utils import to_categorical
from keras import Input
from scipy.io import loadmat
from scipy.misc import imsave

from SVHN.model.ModelA import ModelA
from SVHN.model.ModelB import ModelB
from SVHN.model.ModelC import ModelC

sys.path.append(os.path.join(os.path.dirname(__file__)))


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
import argparse
from keras.datasets import cifar10

from keras.utils import np_utils
from keras.models import load_model


from sklearn.model_selection import train_test_split

CLIP_MAX = 0.5
basedir = os.path.abspath(os.path.dirname(__file__))

def deprocess_image(x):
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape((x.shape[1], x.shape[2], x.shape[3]))  # original shape (1,img_rows, img_cols,1)

def datapredict(dataset='svhn'):

   # dataset == 'svhn':  # 训练集：73257张图像；测试集：26032张图像。
    img_rows, img_cols, img_chn = 32, 32, 3
    input_shape = (img_rows, img_cols, img_chn)
    datasetLoc = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset/') #'../../dataset/'
    train_data = loadmat(datasetLoc + 'train_32x32.mat')
    x_train = np.array(train_data['X'])
    y_train = train_data['y']
    test_data = loadmat(datasetLoc + 'test_32x32.mat')
    x_test = np.array(test_data['X'])
    y_test = test_data['y']

    x_train = np.moveaxis(x_train, -1, 0)
    x_test = np.moveaxis(x_test, -1, 0)
    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train[y_train == 10] = 0
    y_train = np.array(y_train)
    y_test[y_test == 10] = 0
    y_test = np.array(y_test)

    # input image dimensions
    input_tensor = Input(shape=input_shape)

    return x_train, y_train, x_test, y_test,input_tensor

## deepgini
def select_deepgini_old(dataset, model, selectsize, x_test, y_test):

    img_rows, img_cols, img_chn = 32, 32, 3
    x = np.zeros((selectsize, img_rows, img_cols, img_chn))
    y = np.zeros((selectsize,))

    act_layers = model.predict(x_test)
    metrics = np.sum(act_layers ** 2, axis=1)
    rank_lst = np.argsort(metrics)
    index = 0
    for i in range(len(rank_lst)):
        gen_img = x_test[rank_lst[i]].reshape(-1, img_rows, img_cols, img_chn)
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

    x = np.empty(shape=[0, 32, 32, 3])
    y = np.empty(shape=[0, ])


    x_test_array = [[] for i in range(10)]
    y_test_array = [[] for i in range(10)]
    class_nums = int(selectsize // 10)

    act_layers = model.predict(x_test)
    metrics = np.sum(act_layers ** 2, axis=1)
    rank_lst = np.argsort(metrics)

    for i in range(len(rank_lst)):
        gen_img = x_test[rank_lst[i]].reshape(-1, 32, 32, 3)
        pred1 = model.predict(gen_img)
        label1 = np.argmax(pred1[0])
        stop = True
        if label1 == y_test[rank_lst[i]] and len(x_test_array[y_test[rank_lst[i]]]) < class_nums:
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

    x = np.zeros((selectsize, 32, 32, 3))
    y = np.zeros((selectsize,))

    rank_lst = np.random.choice(range(len(x_test)), selectsize, replace=False)

    for i in range(len(rank_lst)):
        x[i] = x_test[rank_lst[i]]
        y[i] = y_test[rank_lst[i]]

    return x, y, rank_lst

if __name__ == "__main__":
    dataset = 'svhn'
    seed_nums = 50
    model_name = 'ModelB'
    x_train, y_train, x_test, y_test, input_tensor = datapredict(dataset)
    y_test = y_test.flatten()
    if model_name == 'none':
        save_dir = './seeds_50_' + 'random' + '/'
    else:
        save_dir = './seeds_50_' + model_name + '/'
        if model_name == 'ModelA':
            model = ModelA(input_tensor=input_tensor)
        elif model_name == 'ModelB':
            model = ModelB(input_tensor=input_tensor)
        elif model_name == 'ModelC':
            model = ModelC(input_tensor=input_tensor)

    if os.path.exists(save_dir):
        for i in os.listdir(save_dir):
            path_file = os.path.join(save_dir, i)
            if os.path.isfile(path_file):
                os.remove(path_file)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if model_name == 'none':
        x_select, y_select, rank_lst = select_random(dataset, seed_nums, x_test, y_test)
    else:
        x_select, y_select, rank_lst = select_deepgini(dataset, model, seed_nums, x_test, y_test)

    print(x_select.shape)
    print(y_select.shape)
    for current_sum in range(seed_nums):
        # below logic is to track number of iterations under progress

        gen_img = np.expand_dims(x_select[current_sum], axis=0)
        gen_img_deprocessed = deprocess_image(gen_img)

        # seed_index = rank_lst[current_sum]
        seed_index = current_sum
        seed_label = int(y_select[current_sum])
        save_img = save_dir + str(seed_index) + '_' + str(seed_label) + '.png'

        imsave(save_img, gen_img_deprocessed)