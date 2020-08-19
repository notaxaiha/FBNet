import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

import torch

from ast import literal_eval
from matplotlib import gridspec

from os.path import join, exists
import os

# prefix_path = '/Data2/home/kysim/Quantization/0602/MY_LOGS'
# prefix_csv_path = join(prefix_path, 'theatas.csv')

class ThetaLoader:
    def __init__(self, architecture_name, search_space_depth=17):
        self.prefix_path = join('./searched_result', architecture_name, 'supernet_function_logs')
        self.search_space_depth=search_space_depth

        # for prior version - theatas.csv
        self.prefix_csv_path = join(self.prefix_path, 'theatas.csv')

        if not os.path.exists(self.prefix_csv_path):
            # for new version - theta.csv
            self.prefix_csv_path = join(self.prefix_path, 'thetas.csv')

        self.prefix_image_path = join(self.prefix_path, 'graph_iamge')

        if not exists(self.prefix_image_path):
            os.makedirs(self.prefix_image_path)


    # get theata and temperature list from csv
    def read_csv(self):
        thetas_csv = pd.read_csv(self.prefix_csv_path)

        self.thetas_list = []
        self.temperature_list = thetas_csv['1'].tolist()
        self.epoch_num = len(thetas_csv)

        for i in range(self.epoch_num): # 180

            # print(thetas_csv['0'].iloc[i])
            temp_theta = literal_eval(thetas_csv['0'].iloc[i])

            self.thetas_list.append(temp_theta)

    def get_softmax_result(self):
        softmax_thetas_list = []

        # (epoch, 22, 9)
        for i in range(self.epoch_num): # 180
            # print(thetas_csv['0'].iloc[i])
            temp_theta = self.thetas_list[i]

            temperature = self.temperature_list[i]

            temp_softmax_list=[]

            for j in range(self.search_space_depth):
                temp_softmax_list.append(torch.nn.functional.softmax(torch.tensor(temp_theta[j])).tolist())
            softmax_thetas_list.append(temp_softmax_list)

        return softmax_thetas_list

    def get_gumbel_softmax_result(self, seed=1):
        #if you want gumbel softmax and static result, you can set like that.
        manual_seed = seed
        np.random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        gumbel_thetas_list = []

        # (epoch, 22, 9)
        for i in range(self.epoch_num): # 180
            # print(thetas_csv['0'].iloc[i])
            temp_theta = self.thetas_list[i]

            temperature = self.temperature_list[i]

            temp_gumbel_list=[]


            for j in range(self.search_space_depth):
                temp_gumbel_list.append(torch.nn.functional.gumbel_softmax(torch.tensor(temp_theta[j]), temperature).tolist())
            gumbel_thetas_list.append(temp_gumbel_list)

        return gumbel_thetas_list


    def plot_architecture_search_trend(self, softmax_thetas_list):
        sample_architecture = np.argmax(softmax_thetas_list,axis=2)
        plt.plot(sample_architecture)
        plt.savefig(join(self.prefix_image_path, 'search_space.jpg'))


    def plot_thetas_on_epoch(self, epoch, save=False):
        plt.plot(self.thetas_list[epoch])

        if save == True:
            plt.savefig(join(self.prefix_image_path, f"thetas_epoch_{epoch}.jpg"))






