'''
usage: python gen_diff.py -h
'''
from __future__ import print_function

import argparse
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定所用的gpu
import warnings
warnings.filterwarnings("ignore")

from scipy.misc import imsave
import time
from model.driving_models import *
from data_utils import load_test_data

from utils_tmp import *


# input image dimensions
img_rows, img_cols = 100, 100
input_shape = (img_rows, img_cols, 3)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

#获取udacity的测试数据
_, _, x_test, y_test = load_test_data()
tmp_list = []
for img in x_test:
    tmp_img = preprocess_image(img)
    tmp_list.append(tmp_img)
x_test = np.array(tmp_list).reshape(-1,100,100,3)
y_test = np.array(y_test)

# load multiple models sharing same input tensor
K.set_learning_phase(0)

# sys.argv = ['4', 0.5, 5, 5, "Dave_dropout"]
neuron_select_strategy = sys.argv[1]
threshold = float(sys.argv[2])
neuron_to_cover_num = int(sys.argv[3])
iteration_times = int(sys.argv[4])
model_name = sys.argv[5]
dataset = 'driving'

if model_name == 'Dave_dropout':
    model1 = Dave_dropout(input_tensor=input_tensor, load_weights=True)
else:
    print('please specify model name')
    os._exit(0)

result_txt = './output/output.txt'
result = 'Running tests for neuron_select_strategy ' + neuron_select_strategy[0] + ' and model ' + model_name + '\n'

print(model1.summary())
# model_layer_dict1 = init_coverage_tables(model1)
model_layer_times1 = init_coverage_times(model1)  # times of each neuron covered
model_layer_times2 = init_coverage_times(model1)  # update when new image and adversarial images found
model_layer_value1 = init_coverage_value(model1)

print(len(model_layer_times2))

# ==============================================================================================
# start gen inputs
if neuron_select_strategy[0] != '4':
    img_dir = 'seed_selection/seeds_50_random'
    save_dir = './generated_inputs_random' + '/'
else:
    img_dir = 'seed_selection/seeds_50_' + model_name
    save_dir = './generated_inputs/' + '/'
if os.path.exists(save_dir):
    for i in os.listdir(save_dir):
        path_file = os.path.join(save_dir, i)
        if os.path.isfile(path_file):
            os.remove(path_file)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

img_paths = os.listdir(img_dir)
img_num = len(img_paths)

pre_num = 10  #预测次数

neuron_to_cover_weight = 0.5
predict_weight = 0.5
learning_step = 2.5#0.02
perturb_adv = 0.02

result += 'Learning_step ' + str(learning_step)  + ' perturb_adversial ' + str(perturb_adv) + '\n'

# start = time.clock()
total_time = 0
total_norm = 0
adversial_num = 0
total_perturb_adversial = 0

for seed_index in range(img_num):
    start_time = time.clock()
    img_list = []
    img_path = os.path.join(img_dir, img_paths[seed_index])

    img_name = img_paths[seed_index].split('.jpg')[0]

    mannual_angle = float(img_name.split('_')[1])

    tmp_img = preprocess_image(img_path)

    orig_img = tmp_img.copy()
    img_list.append(tmp_img)

    update_coverage(tmp_img, model1, model_layer_times2, threshold)

    while len(img_list) > 0:
        gen_img = img_list[0]

        img_list.remove(gen_img)

        # first check if input already induces differences
        # angle1 = model1.predict(gen_img)[0]
        # model1.predict(gen_img):获取到的数据类型为ndarray,shape:(1,1)

        update_coverage(gen_img, model1, model_layer_times1, threshold)

        # 考虑在模型中引入dropout层进行训练,在预测时同样打开dropout层,对一个样本预测T次,取平均值为预测值,计算方差为不确定度

        angle1 = model1.predict(gen_img)[0][0]
        # 注意:model1.predict(gen_img):获取到的数据类型为ndarray,shape:(1,1)

        original_angle1 = angle1
        for pre_n in range(pre_num - 1):
            original_angle1 += model1.predict(gen_img)[0][0]
        original_angle1 = original_angle1 / pre_num  # 计算多次预测值的平均值作为最终预测值

        loss_list = []
        # print(model1.summary())
        # k =model1.get_layer('before_prediction').output
        loss = K.mean(model1.get_layer('before_prediction').output[..., 0])
        loss_mean = loss
        loss_list.append(loss)

        for i in range(pre_num - 1):
            loss = K.mean(model1.get_layer('before_prediction').output[..., 0])
            loss_mean += loss
            loss_list.append(loss)
        loss_mean = loss_mean / pre_num   #计算多次预测值的平均值作为最终预测值

        # loss_con = K.concatenate((a, b), axis=1)

        for i in range(len(loss_list)):
            if i == 0:
                # loss_var = K.square(loss_list[i] - loss_mean) # 计算绝对差值的平方
                loss_var = (loss_list[i] - loss_mean) * (loss_list[i] - loss_mean)
            else:
                loss_var += (loss_list[i] - loss_mean) * (loss_list[i] - loss_mean)
        loss_var = loss_var / len(loss_list)   #计算多次预测值的方差作为衡量该测试数据的不确定度

        layer_output = loss_var

        #神经元选择策略
        loss_neuron,k = neuron_selection(dataset, model1, model_layer_times1, model_layer_value1,
                                       neuron_select_strategy, neuron_to_cover_num,
                                       threshold, model_name, x_test, y_test,False)

        # extreme value means the activation value for a neuron can be as high as possible ...
        EXTREME_VALUE = False
        if EXTREME_VALUE:
            neuron_to_cover_weight = 2  # hyperparameter for balancing two goals

        layer_output += neuron_to_cover_weight * K.sum(loss_neuron)

        # for adversarial image generation
        final_loss = K.mean(layer_output)

        # we compute the gradient of the input picture wrt this loss
        grads = normalize(K.gradients(final_loss, input_tensor)[0])

        grads_tensor_list = loss_list
        grads_tensor_list.extend(loss_neuron)
        grads_tensor_list.append(grads)
        # this function returns the loss and grads given the input picture

        iterate = K.function([input_tensor], grads_tensor_list)

        print("Mutate the seed of number %d" % (seed_index + 1))
        # we run gradient ascent for 3 steps
        for iters in range(iteration_times):

            loss_neuron_list = iterate([gen_img])

            perturb = loss_neuron_list[-1] * learning_step

            gen_img += perturb

            # previous accumulated neuron coverage
            previous_coverage = neuron_covered(model_layer_times1)[2]

            angle1 = model1.predict(gen_img)[0][0]
            # 注意:model1.predict(gen_img):获取到的数据类型为ndarray,shape:(1,1)

            avg_angle1 = angle1

            for pre_n in range(pre_num - 1):
                avg_angle1 += model1.predict(gen_img)[0][0]

            avg_angle1 = avg_angle1 / pre_num  # 计算多次预测值的平均值作为最终预测值


            update_coverage(gen_img, model1, model_layer_times1, threshold)  # for seed selection

            current_coverage = neuron_covered(model_layer_times1)[2]

            diff_img = gen_img - orig_img

            L2_norm = np.linalg.norm(diff_img)

            orig_L2_norm = np.linalg.norm(orig_img)

            perturb_adversial = L2_norm / orig_L2_norm

            if current_coverage - previous_coverage > 0.01 / (i + 1) and perturb_adversial < perturb_adv:
                img_list.append(gen_img)
                # print('coverage diff = ', current_coverage - previous_coverage, 'perturb_adversial = ', perturb_adversial)

            #判断是否为回归任务的对抗样本的条件
            if abs(avg_angle1 - mannual_angle) > 0.2 and not (
                (avg_angle1 > 0 and mannual_angle > 0) or (avg_angle1 < 0 and mannual_angle < 0)
            ):

                print("生成一个对抗样本")
                update_coverage(gen_img, model1, model_layer_times2, threshold)

                total_norm += L2_norm

                total_perturb_adversial += perturb_adversial

                # print('L2 norm : ' + str(L2_norm))
                # print('ratio perturb = ', perturb_adversial)

                gen_img_tmp = gen_img.copy()

                # gen_img_deprocessed = deprocess_image(gen_img_tmp)
                gen_img_deprocessed = draw_arrow(deprocess_image(gen_img_tmp), avg_angle1, mannual_angle)

                save_img = save_dir + img_name + '_' + str(avg_angle1) + '.jpg'

                imsave(save_img, gen_img_deprocessed)

                adversial_num += 1

    end_time = time.clock()

    print('covered neurons percentage %d neurons %.3f'
          % (len(model_layer_times2), neuron_covered(model_layer_times2)[2]))

    duration = end_time - start_time

    print('used time : ' + str(duration))

    total_time += duration

print('k =' + str(k))
result += 'k = ' + str(k) + '\n'
print('All covered neurons percentage %d neurons %.3f'
      % (len(model_layer_times2), neuron_covered(model_layer_times2)[2]))
result += 'All covered neurons percentage ' + str(len(model_layer_times2)) + \
          ' neurons ' + str(neuron_covered(model_layer_times2)[2]) + '\n'

print('total_time = ' + str(total_time))
result += 'total_time = ' + str(total_time) + '\n'

print('total_norm = ' + str(total_norm))
result += 'total_norm = ' + str(total_norm) + '\n'

print('average_norm = ' + str(total_norm / adversial_num))
result += 'average_norm = ' + str(total_norm / adversial_num) + '\n'

print('adversial num = ' + str(adversial_num))
result += 'adversial num = ' + str(adversial_num) + '\n'

print('average perb adversial = ' + str(total_perturb_adversial / adversial_num))
result += 'average perb adversial = ' + str(total_perturb_adversial / adversial_num) + '\n' + '\n'

with open(result_txt, 'a') as file:
    file.write(result)



















