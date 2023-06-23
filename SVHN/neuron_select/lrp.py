# -*- coding: utf-8 -*-
import argparse
import h5py
from datetime import datetime
from keras.models import model_from_json, load_model, save_model
import os,sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'neuron_select'))
print(sys.path)
from utils import load_MNIST, load_CIFAR
from utils import filter_val_set, get_trainable_layers
from utils import generate_adversarial, filter_correct_classifications
from idc import ImportanceDrivenCoverage
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os, sys

__version__ = 0.9

def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
            intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def by_indices(outs, indices):
    return [[outs[i][0][indices]] for i in range(len(outs))]


# 设置参数：version版本、model网络模型、dataset数据集、
#          approach覆盖方法（'idc','nc','kmnc','nbc','snac','tknc','ssc','lsa','dsa'）
#          class类别、layer层次、k_sections、k_neurons、rel_neurons、act_threshold、repeat


# if __name__ == "__main__":
def lrp_neurons(dataset, model,model_name,X_test,Y_test,top_k,Class=True):

    selected_class =  -1  # ALL CLASSES
    repeat =  1
    logfile_name =  'result.log'

    adv_type =  'fgsm'

    logfile = open(logfile_name, 'a')

    ####################
    # 3) Analyze Coverages
    # print("\nRunning IDC for %d relevant neurons" % (num_rel_neurons))

    _, _, X_test_misc, Y_test_misc, = filter_correct_classifications(dataset, model,X_test,Y_test,Class)
    print("Analysed %d test inputs" % len(Y_test_misc))
    #获取与错误的测试样本最相关的神经元以及最不相关的神经元

    if dataset == 'svhn' and len(Y_test_misc) > 300:
        X_test_misc = X_test_misc[:300]
        Y_test_misc = Y_test_misc[:300]
    print("Analysed %d test inputs" % len(Y_test_misc))

    idc = ImportanceDrivenCoverage(dataset, model, model_name, selected_class, X_test_misc, Y_test_misc, Class)
    model_layer_relvance_top_k, p = idc.test(X_test,top_k)

    logfile.close()

    return  model_layer_relvance_top_k, p
