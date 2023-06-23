import numpy as np
from utils import save_totalR, load_totalR
from utils import get_layer_outs_new, create_dir
from lrp_toolbox.model_io import write, read
import os
experiment_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'experiments/')
model_folder      = os.path.join(os.path.dirname(os.path.dirname(__file__)))

class ImportanceDrivenCoverage:
    def __init__(self, dataset, model, model_name, selected_class,
                 train_inputs, train_labels, Class):
        self.covered_combinations = ()
        self.dataset = dataset
        self.model = model
        self.model_name = model_name
        self.selected_class = selected_class
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.model_layer_relvance_top_k = []
        self.p = [] #每个神经元的选取概率，由相关性大小计算
        self.Class = Class

        if self.dataset == 'mnist':
            self.experiment_folder = experiment_folder + 'MNIST'
            self.model_folder = model_folder + '/MNIST'
        elif self.dataset == 'driving':
            self.experiment_folder = experiment_folder + 'Driving'
            self.model_folder = model_folder + '/Driving'
        else:
            self.experiment_folder = experiment_folder + 'SVHN'
            self.model_folder = model_folder + '/SVHN'
        #create experiment directory if not exists
        create_dir(self.experiment_folder)
        create_dir(self.model_folder)


    def get_measure_state(self):
        return self.covered_combinations

    def set_measure_state(self, covered_combinations):
        self.covered_combinations = covered_combinations

    def relvance_rank_weight(self, totalR, top_k):

        model_layer_relvance_dict = {}
        i = 0
        relvance_sum = 0
        for item, layer in enumerate(self.model.layers):
            if item >= len(self.model.layers) - 1: #模型到达最后一层时直接跳出，不执行以下步骤
                break

            if i >= len(totalR):
                break

            if 'input' in layer.name or 'drop' in layer.name or 'flatten' in layer.name:
                continue

            array = []
            for index in range(layer.output_shape[-1]):
                array.append(np.mean(totalR[i][index]))
            i += 1

            if 'flatten' in layer.name: #如果该层为flatten层，则直接跳过，不把它加入相关性分数排序列表中，即不从flatten层中选择神经元来做联合优化
                continue

            index_argsort = np.argsort(-(np.array(array))) #将数组按相关性分数从大到小排序

            for index in index_argsort:
                model_layer_relvance_dict[(layer.name, index)] = array[index]
                if array[index] >= 0:
                    relvance_sum += array[index]

        #对所有神经元的相关性分数排序
        model_layer_relvance_list = sorted(model_layer_relvance_dict.items(), key=lambda x: x[1], reverse=True)

        # 当待排序列表的元素由多字段构成时，我们可以通过sorted(iterable，key，reverse)的参数key来制定我们根据那个字段对列表元素进行排序。
        # iterable -- 可迭代对象
        # key=lambda 元素: 元素[字段索引]
        # 例如：想对元素第二个字段排序，则
        # key=lambda y: y[1] 备注：这里y可以是任意字母，等同key=lambda x: x[1]
        # reverse=True 表示降序
        k = 0
        for (layer_name, index), relvance in model_layer_relvance_list:
            if k >= top_k:
                break
            self.model_layer_relvance_top_k.append((layer_name, index))
            # if relvance >= 0:
            #     self.p.append(relvance / relvance_sum)
            # else:
            #     self.p.append(0)
            k += 1
        # self.p.sort()

        # return self.model_layer_relvance_top_k, self.p

    def relvance_high_top_k(self, totalR, top_k):

        if len(self.model_layer_relvance_top_k) == 0:
            self.relvance_rank_weight(totalR, top_k)  ##根据各个神经元的相关性分数大小，为神经元选择相关性分数最高的前k个神经元


    def test(self, test_inputs,top_k):
        #########################
        #1.Find Relevant Neurons#
        #########################

        # 初始化model中各个层的神经元值为0
        # self.init_coverage_times()

        # print(self.model.layers[1].output_shape[-1])

        print("Loading relevance scores")
        totalR = load_totalR('%s/%s_%s_%d'
                    %(self.experiment_folder, self.model_name,
                        'totalR', self.selected_class), 0)

        if totalR != None:
            self.relvance_high_top_k(totalR,top_k)
            #
            # relevant_neurons = np.argsort(self.totalR[self.subject_layer])[0][::-1][:self.num_relevant_neurons]
            # least_relevant_neurons = np.argsort(self.totalR[self.subject_layer])[0][:self.num_relevant_neurons]

        else:
            print("Relevance scores must be calculated. Doing it now!")
            # Convert keras model into txt
            model_path = self.model_folder + '/' + self.model_name
            write(self.model, model_path, num_channels=test_inputs[0].shape[-1], fmt='keras_txt')

            lrpmodel = read(model_path + '.txt', 'txt')  # 99.16% prediction accuracy
            lrpmodel.drop_softmax_output_layer()  # drop softmax output layer for analysis

            totalR = find_relevant_neurons(
                self.model, lrpmodel, self.train_inputs, self.train_labels, None, 'sum')

            for index in range(len(totalR)):
                if np.array(totalR[index]).ndim == 4:
                    (n, h, w, num) = np.array(totalR[index]).shape
                    totalR[index] = totalR[index].reshape(num, n, h, w)
                else:
                    (n, num) = np.array(totalR[index]).shape
                    totalR[index] = totalR[index].reshape(num, n)

            save_totalR(totalR, '%s/%s_%s_%d'
                        %(self.experiment_folder, self.model_name,
                          'totalR', self.selected_class), 0)

            self.relvance_high_top_k(totalR, top_k)

        # return relevant_neurons, least_relevant_neurons
        return self.model_layer_relvance_top_k, self.p

    ###################end：自己添加和修改的代码部分#######################


def find_relevant_neurons(kerasmodel, lrpmodel, inps, outs, lrpmethod=None, final_relevance_method='sum'):

    #final_relevants = np.zeros([1, kerasmodel.layers[subject_layer].output_shape[-1]])

    totalR = None
    cnt = 0
    for inp in inps:
        cnt+=1
        ypred = lrpmodel.forward(np.expand_dims(inp, axis=0))
        #prepare initial relevance to reflect the model's dominant prediction (ie depopulate non-dominant output neurons)
        mask = np.zeros_like(ypred)
        mask[:,np.argmax(ypred)] = 1
        Rinit = ypred*mask

        if not lrpmethod:
            R_inp, R_all = lrpmodel.lrp(Rinit)                   #as Eq(56) from DOI: 10.1371/journal.pone.0130140
        elif lrpmethod == 'epsilon':
            R_inp, R_all = lrpmodel.lrp(Rinit,'epsilon',0.01)    #as Eq(58) from DOI: 10.1371/journal.pone.0130140
        elif lrpmethod == 'alphabeta':
            R_inp, R_all = lrpmodel.lrp(Rinit,'alphabeta',3)     #as Eq(60) from DOI: 10.1371/journal.pone.0130140
        else:
            print('Unknown LRP method!')
            raise Exception

        if totalR:
            for idx, elem in enumerate(totalR):
                totalR[idx] = elem + R_all[idx]
        else:
            totalR = R_all

    #      THE MOST RELEVANT                               THE LEAST RELEVANT
    # return np.argsort(totalR[subject_layer])[0][::-1][:num_rel], np.argsort(totalR[subject_layer])[0][:num_rel], totalR
    return totalR




