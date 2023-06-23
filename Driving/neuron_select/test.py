import argparse
import h5py
from datetime import datetime
from keras.models import model_from_json, load_model, save_model
from neuron_select.utils import load_MNIST, load_CIFAR
from neuron_select.utils import filter_val_set, get_trainable_layers
from neuron_select.utils import generate_adversarial, filter_correct_classifications
from SVHN.neuron_select.idc import ImportanceDrivenCoverage
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
def relevant_neurons(model,model_name,X_test,Y_test):
    # 加载参数

    num_rel_neurons =  20
    act_threshold = 0.5

    selected_class =  -1  # ALL CLASSES
    repeat =  1
    logfile_name =  'result.log'

    adv_type =  'fgsm'

    logfile = open(logfile_name, 'a')


    # 2) Load necessary information
    trainable_layers = get_trainable_layers(model)  # 除去输入层，softmax层等，将其他层加入训练层
    non_trainable_layers = list(set(range(len(model.layers))) - set(trainable_layers))
    print('Trainable layers: ' + str(trainable_layers))
    print('Non trainable layers: ' + str(non_trainable_layers))

    experiment_folder = 'experiments'


    #从这里开始看代码

    # Investigate the penultimate layer  调查倒数第二层（作为目标层）
    subject_layer = -1  #subject_layer：-1表示最后一层
    subject_layer = trainable_layers[subject_layer]

    ####################
    # 3) Analyze Coverages
    print("\nRunning IDC for %d relevant neurons" % (num_rel_neurons))

    _, _, X_test_misc, Y_test_misc, = filter_correct_classifications(model,
                                                                        X_test,
                                                                        Y_test)
    #获取最相关的神经元以及最不相关的神经元

    idc = ImportanceDrivenCoverage(model, model_name, num_rel_neurons, selected_class,
                                       subject_layer, X_test_misc, Y_test_misc)

    relevant_neurons = idc.test(X_test)

    print("Analysed %d test inputs" % len(Y_test_misc))

    logfile.close()
    return  subject_layer, relevant_neurons
