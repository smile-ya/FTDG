# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:40:17 2019

@author: qq
"""
import math
import os

from keras import Input
from scipy.misc import imsave

from Driving.data_utils import load_test_data

from Driving.model.driving_models import Dave_dropout

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
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

CLIP_MAX = 0.5
basedir = os.path.abspath(os.path.dirname(__file__))

def preprocess_image(img_path, target_size=(100, 100)):
    img = image.load_img(img_path, target_size=target_size)
    input_img_data = image.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)
    return input_img_data

def deprocess_image(x):
    x = x.reshape((100, 100, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def datapredict(dataset='driving'):
    img_rows, img_cols = 100, 100
    input_shape = (img_rows, img_cols, 3)
    _, _, x_test, y_test = load_test_data()

    input_tensor = Input(shape=input_shape)

    return x_test, y_test,input_tensor



#
# def select_var_old(model, predict_times, seed_nums, d, x_test, y_test, img_path):
#
#     # x = np.zeros((0, 100, 100, 3))
#     # y = np.zeros((0,))
#     x = np.empty(shape=[0, 100, 100, 3])
#     y = np.empty(shape=[0,])
#     select_path = np.array([])
#     img_path = np.array(img_path)
#     region_nums = math.ceil((np.max(y_test) - np.min(y_test)) / d)
#
#     region_array = [[] for i in range(region_nums)]
#     x_test_array = [[] for i in range(region_nums)]
#     y_test_array = [[] for i in range(region_nums)]
#     rank_lst = [[] for i in range(region_nums)]
#
#     for i in range(len(y_test)):
#         index = int((y_test[i] - np.min(y_test)) // d)
#         region_array[index].append(i)
#
#     for i in range(len(region_array)):
#         x_test_array[i] = x_test[region_array[i]]
#         y_test_array[i] = y_test[region_array[i]]
#
#     for i in range(len(x_test_array)):
#         pre = model.predict(x_test_array[i])
#         pre1 = model.predict(x_test_array[i])
#         out = np.hstack((pre, pre1))  # 水平拼接
#         for pre_num in range(predict_times-2):
#             pre_iter = model.predict(x_test_array[i])
#             out = np.hstack((out, pre_iter))  # 水平拼接
#
#         mean = np.mean(out, axis=1) #按行计算平均值,平均值即为预测值
#         var = np.var(out, axis=1)   #按行计算方差
#         rank_lst[i] = np.argsort(-var)
#         for j in range(len(rank_lst[i])):
#             if not(abs(mean[rank_lst[i][j]] - y_test[rank_lst[i][j]]) < 0.2 \
#                     and ((mean[rank_lst[i][j]] > 0 and y_test[rank_lst[i][j]] > 0) or (
#                     mean[rank_lst[i][j]] < 0 and y_test[rank_lst[i][j]] < 0))):
#
#                 rank_lst[i] = np.delete(rank_lst[i],j)
#
#     region_seeds = int(seed_nums // region_nums)
#     region_more = int(seed_nums % region_nums) #取余数
#     for i in range(len(rank_lst)): #len(rank_lst)等于region_nums
#         if i == region_nums-1:
#             x = np.concatenate((x,x_test[rank_lst[i][0:region_seeds+region_more]]))
#             y = np.concatenate((y,y_test[rank_lst[i][0:region_seeds + region_more]]))
#             # x.append(x_test[rank_lst[i][0:region_seeds+region_more]])
#             # y.append(y_test[rank_lst[i][0:region_seeds+region_more]])
#             select_path = np.append(select_path, img_path[rank_lst[i][0:region_seeds+region_more]])
#         else:
#             x = np.concatenate((x, x_test[rank_lst[i][0:region_seeds]]))
#             y = np.concatenate((y, y_test[rank_lst[i][0:region_seeds]]))
#             # x.append(x_test[rank_lst[i][0:region_seeds]])
#             # y.append(y_test[rank_lst[i][0:region_seeds]])
#             select_path = np.append(select_path, img_path[rank_lst[i][0:region_seeds]])
#     return x, y, select_path

#

# 计算各个测试样本的方差
def select_var(model, predict_times, seed_nums, x_test, y_test, img_path):

    x = np.zeros((seed_nums, 100, 100, 3))
    y = np.zeros((seed_nums,))
    select_path = []

    pre = model.predict(x_test)
    pre1 = model.predict(x_test)
    out = np.hstack((pre, pre1))  # 水平拼接
    for pre_num in range(predict_times-2):
        pre_iter = model.predict(x_test)
        out = np.hstack((out, pre_iter))  # 水平拼接

    mean = np.mean(out, axis=1) #按行计算平均值,平均值即为预测值
    var = np.var(out, axis=1)   #按行计算方差

    rank_lst = np.argsort(-var)
    index = 0
    for i in range(len(rank_lst)):
        # if abs( mean[rank_lst[i]] - y_test[rank_lst[i]] ) < 0.2 \
        #         and ( (mean[rank_lst[i]] > 0 and y_test[rank_lst[i]] > 0) or (mean[rank_lst[i]] < 0 and y_test[rank_lst[i]] < 0) ) \
        #         and index < seed_nums:
        x[index] = x_test[rank_lst[i]]
        y[index] = y_test[rank_lst[i]]
        select_path.append(img_path[rank_lst[i]])
        index += 1
        if index >= seed_nums:
            break

    return x, y, select_path


# def select_var(model, predict_times, seed_nums, d, x_test, y_test, img_path):
#
#     # x = np.zeros((0, 100, 100, 3))
#     # y = np.zeros((0,))
#     x = np.empty(shape=[0, 100, 100, 3])
#     y = np.empty(shape=[0,])
#     select_path = np.array([])
#     img_path = np.array(img_path)
#     region_nums = math.ceil((np.max(y_test) - np.min(y_test)) / d)
#
#     region_array = [[] for i in range(region_nums)]
#     x_test_array = [[] for i in range(region_nums)]
#     y_test_array = [[] for i in range(region_nums)]
#     rank_lst = [[] for i in range(region_nums)]
#
#     for i in range(len(y_test)):
#         index = int((y_test[i] - np.min(y_test)) // d)
#         region_array[index].append(i)
#
#     for i in range(len(region_array)):
#         x_test_array[i] = x_test[region_array[i]]
#         y_test_array[i] = y_test[region_array[i]]
#
#     for i in range(len(x_test_array)):
#         pre = model.predict(x_test_array[i])
#         pre1 = model.predict(x_test_array[i])
#         out = np.hstack((pre, pre1))  # 水平拼接
#         for pre_num in range(predict_times-2):
#             pre_iter = model.predict(x_test_array[i])
#             out = np.hstack((out, pre_iter))  # 水平拼接
#
#         mean = np.mean(out, axis=1) #按行计算平均值,平均值即为预测值
#         var = np.var(out, axis=1)   #按行计算方差
#         rank_lst[i] = np.argsort(-var)
#
#         #删除测试数据中的对抗样本，即触发模型错误的样本
#         # for j in range(len(rank_lst[i])):
#         #     if not(abs(mean[rank_lst[i][j]] - y_test[rank_lst[i][j]]) < 0.2 \
#         #             and ((mean[rank_lst[i][j]] > 0 and y_test[rank_lst[i][j]] > 0) or (
#         #             mean[rank_lst[i][j]] < 0 and y_test[rank_lst[i][j]] < 0))):
#         #
#         #         rank_lst[i] = np.delete(rank_lst[i],j)
#
#     region_seeds = int(seed_nums // region_nums)
#     region_more = int(seed_nums % region_nums) #取余数
#     for i in range(len(rank_lst)): #len(rank_lst)等于region_nums
#         if region_more > 0:
#             x = np.concatenate((x, x_test[rank_lst[i][0:region_seeds+1]]))
#             y = np.concatenate((y, y_test[rank_lst[i][0:region_seeds+1]]))
#             select_path = np.append(select_path, img_path[rank_lst[i][0:region_seeds+1]])
#             # index = np.arange(region_seeds+1)
#             # rank_lst[i] = np.delete(rank_lst[i], index)
#             region_more = region_more - 1
#         else:
#             x = np.concatenate((x, x_test[rank_lst[i][0:region_seeds]]))
#             y = np.concatenate((y, y_test[rank_lst[i][0:region_seeds]]))
#             select_path = np.append(select_path, img_path[rank_lst[i][0:region_seeds]])
#             # index = np.arange(region_seeds)
#             # rank_lst[i] = np.delete(rank_lst[i], index)
#
#     return x, y, select_path


def select_random(selectsize, x_test, y_test,img_path):
    x = np.zeros((selectsize, 100, 100, 3))
    y = np.zeros((selectsize,))
    select_path = []
    rank_lst = np.random.choice(range(len(x_test)), selectsize, replace=False)
    for i in range(len(rank_lst)):
        x[i] = x_test[rank_lst[i]]
        y[i] = y_test[rank_lst[i]]
        select_path.append(img_path[rank_lst[i]])
    return x, y, select_path

if __name__ == "__main__":
    dataset = 'mnist'
    seed_nums = 50
    model_name = 'Dave_dropout'
    predict_times = 20
    x_test, y_test, input_tensor = datapredict()
    if model_name == 'none':
        save_dir = './seeds_50_' + 'random' + '/'
    else:
        # save_dir = './seeds_50_' + model_name + '/'
        save_dir = './seeds_50_' + model_name + '/'
        model1 = Dave_dropout(input_tensor=input_tensor, load_weights=True)

    if os.path.exists(save_dir):
        for i in os.listdir(save_dir):
            path_file = os.path.join(save_dir, i)
            if os.path.isfile(path_file):
                os.remove(path_file)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_path = x_test
    tmp_list = []
    for img in x_test:
        tmp_img = preprocess_image(img)
        tmp_list.append(tmp_img)
    x_test = np.array(tmp_list).reshape(-1,100,100,3)
    y_test = np.array(y_test)

    if model_name == 'none':
        x_select, y_select, select_path = select_random(seed_nums, x_test, y_test,img_path)
    else:
        x_select, y_select, select_path = select_var(model1, predict_times, seed_nums, x_test, y_test, img_path)


    print(x_select.shape)
    print(y_select.shape)

    for current_sum in range(seed_nums):
        # below logic is to track number of iterations under progress

        gen_img = np.expand_dims(x_select[current_sum], axis=0) #扩展维度，按列扩展 shape = (1, 100, 100, 3)
        gen_img_deprocessed = deprocess_image(gen_img)

        # seed_index = select_path[current_sum].split('/')[-1].split('.')[0]
        seed_index = current_sum
        seed_angle = y_select[current_sum]
        save_img = save_dir + str(seed_index) + '_' + str(seed_angle) + '.jpg'

        imsave(save_img, gen_img_deprocessed)


