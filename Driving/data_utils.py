
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

import time
from configs import bcolors
from utils_tmp import *

training_data = os.path.join(os.path.dirname(__file__),'hmb/training/')
testing_data = os.path.join(os.path.dirname(__file__),'hmb/testing/')

def preprocess(path, target_size):
    return preprocess_image(path, target_size)[0]

def data_generator(xs, ys, target_size, batch_size=64):
    gen_state = 0
    while 1:
        if gen_state + batch_size > len(xs):
            paths = xs[gen_state: len(xs)]
            y = ys[gen_state: len(xs)]
            X = [preprocess(x, target_size) for x in paths]
            gen_state = 0
        else:
            paths = xs[gen_state: gen_state + batch_size]
            y = ys[gen_state: gen_state + batch_size]
            X = [preprocess(x, target_size) for x in paths]
            gen_state += batch_size
        yield np.array(X), np.array(y)

def load_train_data(path=training_data, batch_size=64, shape=(100, 100)):
    xs = []
    ys = []
    start_load_time = time.time()
    with open(path + 'trainingall_steering.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            # xs.append(path + line.split(',')[5])
            # ys.append(float(line.split(',')[6]))
            xs.append(path + 'center/' + line.split(',')[0] + '.jpg')
            ys.append(float(line.split(',')[1]))
    # shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)

    train_xs = xs
    train_ys = ys

    train_generator = data_generator(train_xs, train_ys,
                                     target_size=(shape[0], shape[1]),
                                     batch_size=batch_size)

    print(bcolors.OKBLUE + 'finished loading data, running time: {} seconds'.format(
        time.time() - start_load_time) + bcolors.ENDC)
    return train_generator, len(train_xs)

def load_test_data(path=testing_data, batch_size=64, shape=(100, 100)):
    xs = []
    ys = []
    start_load_time = time.time()
    with open(path + 'hmb3_steering.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            xs.append(path + 'center/' + line.split(',')[0] + '.jpg')
            ys.append(float(line.split(',')[1]))
    # shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)

    train_xs = xs
    train_ys = ys

    train_generator = data_generator(train_xs, train_ys,
                                     target_size=(shape[0], shape[1]),
                                     batch_size=batch_size)

    print(bcolors.OKBLUE + 'finished loading data, running time: {} seconds'.format(
        time.time() - start_load_time) + bcolors.ENDC)
    return train_generator, len(train_xs),train_xs,train_ys
