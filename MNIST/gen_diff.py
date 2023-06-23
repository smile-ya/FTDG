# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # 指定所用的gpu
import warnings
warnings.filterwarnings("ignore")
from keras.datasets import mnist
from keras.layers import Input
from keras.utils import to_categorical
from scipy.misc import imsave
from utils_tmp import *
import os
import time
from MNIST.model.Model1 import Model1
from MNIST.model.Model2 import Model2
from MNIST.model.Model3 import Model3
from MNIST.model.Model4 import Model4

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
K.set_learning_phase(0)

# sys.argv = ['4', 0.5, 5, 5, "Model1"]
neuron_select_strategy = sys.argv[1]
threshold = float(sys.argv[2])
neuron_to_cover_num = int(sys.argv[3])
iteration_times = int(sys.argv[4])
model_name = sys.argv[5]
dataset = 'mnist'
neuron_to_cover_weight = 0.5
predict_weight = 0.5
learning_step = 0.02
perturb_adv = 0.02

if model_name == 'Model1':
    model1 = Model1(input_tensor=input_tensor)
elif model_name == 'Model2':
    model1 = Model2(input_tensor=input_tensor)
elif model_name == 'Model3':
    model1 = Model3(input_tensor=input_tensor)
elif model_name == 'Model4':
    model1 = Model4(input_tensor=input_tensor)
else:
    print('please specify model name')
    os._exit(0)

result_txt = './output/output.txt'
result = 'Running testss for neuron_select_strategy ' + neuron_select_strategy[0] + ' and model ' + model_name + '\n'

print(model1.summary())
# model_layer_dict1 = init_coverage_tables(model1)
model_layer_times1 = init_coverage_times(model1)  # times of each neuron covered
model_layer_times2 = init_coverage_times(model1)  # update when new image and adversarial images found
model_layer_value1 = init_coverage_value(model1)

# relvance_model_layer_times3 = relvance_init_coverage_times(model1) #对计算了相关性分数的网络层的神经元进行初始化

# start gen inputs
# img_paths = image.list_pictures('../seeds_20', ext='JPEG')
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

# e.g.[0,1,2] None for neurons not covered, 0 for covered often, 1 for covered rarely, 2 for high weights


result += 'Learning_step ' + str(learning_step)  + ' perturb_adversial ' + str(perturb_adv) + '\n'

# start = time.clock()
total_time = 0
total_norm = 0
adversial_num = 0

total_perturb_adversial = 0

for i in range(img_num):

    start_time = time.clock()

    img_list = []

    img_path = os.path.join(img_dir,img_paths[i])

    img_name = img_paths[i].split('.')[0]

    mannual_label = int(img_name.split('_')[1])

    # print(img_path)

    tmp_img = preprocess_image(img_path)

    orig_img = tmp_img.copy()

    img_list.append(tmp_img)

    update_coverage(tmp_img, model1, model_layer_times2, threshold)

    while len(img_list) > 0:

        gen_img = img_list[0]

        img_list.remove(gen_img)

        # first check if input already induces differences
        pred1 = model1.predict(gen_img)
        label1 = np.argmax(pred1[0])

        label_top5 = np.argsort(pred1[0])[-5:]

        # update_coverage_value(gen_img, model1, model_layer_value1)
        update_coverage(gen_img, model1, model_layer_times1, threshold)

        orig_label = label1
        orig_pred = pred1

        loss_1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss_2 = K.mean(model1.get_layer('before_softmax').output[..., label_top5[-2]])
        loss_3 = K.mean(model1.get_layer('before_softmax').output[..., label_top5[-3]])
        loss_4 = K.mean(model1.get_layer('before_softmax').output[..., label_top5[-4]])
        loss_5 = K.mean(model1.get_layer('before_softmax').output[..., label_top5[-5]])

        layer_output = (predict_weight * (loss_2 + loss_3 + loss_4 + loss_5) - loss_1)

        # neuron coverage loss 神经元选择策略
        loss_neuron = neuron_selection(dataset, model1, model_layer_times1, model_layer_value1,
                                       neuron_select_strategy,neuron_to_cover_num,
                                       threshold,model_name,x_test,y_test,True)
        # loss_neuron = neuron_scale(loss_neuron) # useless, and negative result

        # extreme value means the activation value for a neuron can be as high as possible ...
        EXTREME_VALUE = False
        if EXTREME_VALUE:
            neuron_to_cover_weight = 2  #hyperparameter for balancing two goals

        layer_output += neuron_to_cover_weight * K.sum(loss_neuron)

        # for adversarial image generation
        final_loss = K.mean(layer_output)

        # we compute the gradient of the input picture wrt this loss
        grads = normalize(K.gradients(final_loss, input_tensor)[0])

        grads_tensor_list = [loss_1, loss_2, loss_3, loss_4, loss_5]
        grads_tensor_list.extend(loss_neuron)
        grads_tensor_list.append(grads)
        # this function returns the loss and grads given the input picture

        iterate = K.function([input_tensor], grads_tensor_list)

        print("Mutate the seed of number %d" % (i + 1))

        # we run gradient ascent for 3 steps
        for iters in range(iteration_times):

            loss_neuron_list = iterate([gen_img])

            perturb = loss_neuron_list[-1] * learning_step

            gen_img += perturb

            # previous accumulated neuron coverage
            previous_coverage = neuron_covered(model_layer_times1)[2]

            pred1 = model1.predict(gen_img)
            label1 = np.argmax(pred1[0])

            update_coverage(gen_img, model1, model_layer_times1, threshold) # for seed selection

            current_coverage = neuron_covered(model_layer_times1)[2]

            diff_img = gen_img - orig_img

            L2_norm = np.linalg.norm(diff_img)

            orig_L2_norm = np.linalg.norm(orig_img)

            perturb_adversial = L2_norm / orig_L2_norm

            if current_coverage - previous_coverage > 0.01 / (i + 1) and perturb_adversial < perturb_adv:
                img_list.append(gen_img)
                # print('coverage diff = ', current_coverage - previous_coverage, 'perturb_adversial = ', perturb_adversial)

            if label1 != orig_label:
                update_coverage(gen_img, model1, model_layer_times2, threshold)

                total_norm += L2_norm

                total_perturb_adversial += perturb_adversial

                # print('L2 norm : ' + str(L2_norm))
                # print('ratio perturb = ', perturb_adversial)

                gen_img_tmp = gen_img.copy()

                gen_img_deprocessed = deprocess_image(gen_img_tmp)

                save_img = save_dir + img_name + '_' + str(label1) + '.png'
                print(save_img)
                imsave(save_img, gen_img_deprocessed)

                adversial_num += 1

    end_time = time.clock()


    print('covered neurons percentage %d neurons %.3f'
          % (len(model_layer_times2), neuron_covered(model_layer_times2)[2]))

    duration = end_time - start_time

    print('used time : ' + str(duration))

    total_time += duration


print('All covered neurons percentage %d neurons %.3f'
      % (len(model_layer_times2), neuron_covered(model_layer_times2)[2]))
result += 'All covered neurons percentage ' + str(len(model_layer_times2)) +\
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