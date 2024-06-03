import math

import numpy

import pandas as pd

from universal.algo import Algo
from universal.result import ListResult
import numpy as np

from universal import tools

import os

import csv
import heapq


class AT_ucb(Algo):
    """ Bay and hold strategy. Buy equal amount of each stock in the beginning and hold them
        forever.  """
    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self, datasetname, epsilon, window=5, eps=10, percentage=0.5, ndays=None, expectationReturn=None,
                 tau=3, d = 4):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """

        super(AT_ucb, self).__init__(min_history=window)

        # input check
        if window < 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')

        self.window = window
        self.eps = eps

        self.d = d
        self.histLen = 0  # yjf.
        self.datasetname = datasetname
        self.percentage = percentage
        self.batch = window
        self.history = None
        self.ndays = ndays
        self.expectationReturn = expectationReturn
        self.dropTop = []
        self.dropLow = []
        self.epsilon = epsilon
        self.tau = tau
        self.data = tools.dataset(datasetname)
        # self.kList = [50, 100, 1000, 20]
        # self.kList = [20, 50, 100, 1000, 0.05, 0.5, 5, 10, 15]       # djia
        # self.kList = [20, 50, 100, 1000, 10, 15]  # sp500
        self.kList = [5, 50, 15]  # ******
        # self.kList = [15, 50, 1000]  # ******
        # self.kList = [5]  #
        self.lamda = 1
        # self.wkList = [2, 5, 7]  # **********
        self.wkList = [ 3, 4, 5]
        # self.wkList = [ 4, 5, 6]
        self.sList = [0] * len(self.kList)
        self.NList = [0] * len(self.kList)
        self.wsList = [0] * len(self.wkList)
        self.wNList = [0] * len(self.wkList)
        # self.sklist(self.datasetname)
        self.alpha = 2
        self.beta = 2
        self.timeGap = 0

    def init_weights(self, m):
        return np.ones(m) / m

    def addToCsv(self, vec, minpath):

        path = os.getcwd() + '/picture/' + minpath + '_model.csv'
        file = open(path, 'a')
        csv_write = csv.writer(file)
        csv_write.writerow(vec)

    def step(self, x, last_b, history):  # cgy new
        """
        :param x: the last row data of history
        :param last_b:
        :param history:
        :return:
        """

        self.histLen = history.shape[0]

        # self.tau = self.ucb_algo_w(x, last_b)

        # if self.histLen % 10 == 0:
        #     self.tau = self.ucb_algo_w(x, last_b)

        if self.timeGap % self.d == 0:  # and self.histLen != 25:  # & self.histLen != 0

            wklistEachCW = []

            for i in range(len(self.wkList)):
                wklistEachCW.append(np.dot(self.my_step(history, self.wkList[i]), x))

            idx = np.argmin(wklistEachCW)
            # max_indices = np.argsort(wklistEachCW)[-2:]
            # idx = max_indices[0] if wklistEachCW[max_indices[0]] != wklistEachCW[max_indices[1]] else max_indices[1]
            self.tau = self.wkList[idx]
            # print(self.tau, self.histLen)
            self.timeGap = 0

        # self.sma = (history.iloc[-2] + history.iloc[-3] + history.iloc[-4] + history.iloc[-5]) / 4
        else:
            self.timeGap = self.timeGap + 1
        self.ema = (history.iloc[-2] * 0.9)

        last_window_history = history[-self.tau:]

        # self.epsilon = 50

        max_p = self.calculate_max_p(last_window_history)

        # if self.histLen % 100 == 0:
        self.epsilon = self.ucb_algo_h(x, last_b, history)
        # self.epsilon = self.ucb_algo(x, last_b)

        last_b = history.iloc[-1] / self.ema

        max_p_t = self.calculate_max_p_T(max_p, last_window_history.shape[1])
        b = self.update_new_b(max_p, max_p_t, last_b)
        b = tools.simplex_proj(b)

        return b

    def my_step(self, history, tau):  # cgy new
        """
        :param x: the last row data of history
        :param last_b:
        :param history:
        :return:
        """

        # self.tau = self.ucb_algo_w(x, last_b)

        # if self.histLen % 10 == 0:
        #     self.tau = self.ucb_algo_w(x, last_b)

        last_window_history = history[-tau:]

        max_p = self.calculate_max_p(last_window_history)

        ema = (history.iloc[-2] * 0.9)

        last_b = history.iloc[-1] / ema

        max_p_t = self.calculate_max_p_T(max_p, last_window_history.shape[1])
        b = self.update_new_b(max_p, max_p_t, last_b)
        b = tools.simplex_proj(b)

        return b

    # def step(self, x, last_b, history): #tjh old
    #     """
    #
    #     :param x: the last row data of history
    #     :param last_b:
    #     :param history:
    #     :return:
    #     """
    #
    #     self.histLen = history.shape[0]
    #
    #     last_window_history = history[-self.tau:]
    #
    #     self.epsilon = 50
    #
    #     max_p = self.calculate_max_p(last_window_history)
    #
    #     last_b = history.iloc[-1] / history.iloc[-2]
    #
    #     max_p_t = self.calculate_max_p_T(max_p, last_window_history.shape[1])
    #     b = self.update_new_b(max_p, max_p_t, last_b)
    #     b = tools.simplex_proj(b)
    #
    #     return b

    # def cal_w(self, history):
    #     if

    def calculate_max_p(self, last_window_history):
        """
        calculate max price of nearly history
        :param history: price sequence of subset
        :param tau: time windows
        :return:
        """

        return last_window_history.max() / last_window_history.iloc[-1]  #

    def calculate_max_p_T(self, max_p, asset_amount):
        unit_vector = np.ones(max_p.shape[0])
        max_p_t = max_p - (np.dot(unit_vector, max_p) / asset_amount) * unit_vector
        return max_p_t

    def update_new_b(self, max_p, max_p_t, last_b_k):  # cgy new

        condition1 = np.dot(max_p, max_p_t)  #
        condition2 = last_b_k + (self.epsilon * max_p_t) / (condition1 ** 0.5)  # b^的值
        if condition1 == 0:
            # print(0)
            return last_b_k
        elif condition2.min() < 0:
            gamma = (last_b_k / (last_b_k - condition2)).max()
            # k = map(list(max_p).index, heapq.nlargest(1, list(max_p)))
            # k = list(k)
            # k = k[0]
            # for i in range(max_p.shape[0]):
            #     if i == k:
            #         max_p[i] = 1
            #     else:
            #         max_p[i] = 0

            # last_b_k = (1 - gamma) * condition2 + gamma * last_b_k # cgy
            last_b_k = (1 - gamma) * last_b_k + gamma * condition2
            # last_b_k = max_p
            # print(1)
        else:
            last_b_k = last_b_k + (self.epsilon * max_p_t) / (condition1 ** 0.5)
            # print(2)
        return last_b_k

    # def update_new_b(self, max_p, max_p_t, last_b_k): #tjh old
    #
    #     condition1 = np.dot(max_p, max_p_t) #
    #     condition2 = last_b_k + (self.epsilon * max_p_t) / (condition1 ** 0.5) # b^的值
    #     if condition1 == 0:
    #         return last_b_k
    #     elif condition2.min() < 0:
    #         k = map(list(max_p).index, heapq.nlargest(1, list(max_p)))
    #         k = list(k)
    #         k = k[0]
    #         for i in range(max_p.shape[0]):
    #             if i == k:
    #                 max_p[i] = 1
    #             else:
    #                 max_p[i] = 0
    #
    #         last_b_k = max_p
    #     else:
    #         last_b_k = last_b_k + (self.epsilon * max_p_t) / (condition1 ** 0.5)
    #     return last_b_k

    # def sklist(self, d):
    #     datasets = ['nyse_o',  'hs300', 'nyse_n','sp500','djia', 'tse', 'nasdaq(m+)', 'nasdaq(s-)']
    #     if d in [datasets[:2]]:
    #         self.tau = 4
    #     elif d in [datasets[5:]]:
    #         self.tau = 3
    #     elif d in [datasets[3:5]]:
    #         self.tau = 6
    #     else:
    #         self.tau = 5

    def ucb_algo_old(self, x, last_b):

        m = len(self.kList)
        t = self.histLen
        s = np.dot(last_b, x)
        sumList = []
        for i in range(m):
            # k = self.kList[i]
            s_k = (self.sList[i] * self.kList[i] + s) / (self.kList[i] + 1)
            N_k = self.NList[i] + 1
            CI_k = ((1 + N_k) / N_k ** 2 * (1 + 2 * math.log(m * (1 + N_k) ** 0.5 * t))) ** 0.5
            sum = s_k + CI_k
            sumList.append(sum)
        index_max = np.argmax(sumList)
        self.sList[index_max] = (self.sList[index_max] * self.kList[index_max] + s) / (self.kList[index_max] + 1)
        self.NList[index_max] = self.NList[index_max] + 1
        k = self.kList[index_max]

        if self.histLen == self.data.shape[0]:
            print('the times of k:', self.NList)
        return k

    def ucb_algo(self, x, last_b):

        m = len(self.kList)
        t = self.histLen
        s = np.dot(last_b, x)
        sumList = []
        for i in range(m):
            # k = self.kList[i]
            # s_k = (self.sList[i] * self.kList[i] + s) / (self.kList[i] + 1)
            s_k = (self.sList[i] * self.NList[i] + s) / (self.NList[i] + 1)
            N_k = self.NList[i] + 1
            CI_k = ((1 + N_k) / N_k ** 2 * (1 + 2 * math.log(m * (1 + N_k) ** 0.5 * t))) ** 0.5
            sum = s_k + CI_k
            sumList.append(sum)
        index_max = np.argmax(sumList)
        self.sList[index_max] = (self.sList[index_max] * self.NList[index_max] + s) / (self.NList[index_max] + 1)
        self.NList[index_max] = self.NList[index_max] + 1
        k = self.kList[index_max]

        if self.histLen == self.data.shape[0]:
            print('the times of k:', self.NList)
        return k

    def ucb_algo_h(self, x, last_b, history):

        m = len(self.kList)
        t = self.histLen
        x_t = history.iloc[-1] / history.iloc[-2]
        s = np.dot(last_b, x_t)
        sumList = []
        for i in range(m):
            # k = self.kList[i]
            s_k = (self.sList[i] * self.kList[i] + s) / (self.kList[i] + 1)
            N_k = self.NList[i] + 1
            CI_k = ((1 + N_k) / N_k ** 2 * (1 + 2 * math.log(m * (1 + N_k) ** 0.5 * t))) ** 0.5
            sum = s_k + CI_k
            sumList.append(sum)
        index_max = np.argmax(sumList)
        self.sList[index_max] = (self.sList[index_max] * self.kList[index_max] + s) / (self.kList[index_max] + 1)
        self.NList[index_max] = self.NList[index_max] + 1
        k = self.kList[index_max]

        if self.histLen == self.data.shape[0]:
            print('the times of k:', self.NList)
        return k

    def ucb_algo_w(self, x, last_b):

        m = len(self.wkList)
        t = self.histLen
        s = np.dot(last_b, x)
        # s = np.dot(last_b, pt/ pt-1)

        sumList = []
        for i in range(m):
            # k = self.kList[i]
            s_k = (self.wsList[i] * self.wkList[i] + s) / (self.wkList[i] + 1)
            N_k = self.wNList[i] + 1
            CI_k = ((1 + N_k) / N_k ** 2 * (1 + 2 * math.log(m * (1 + N_k) ** 0.5 * t))) ** 0.5
            sum = s_k + CI_k
            sumList.append(sum)
        index_max = np.argmax(sumList)
        self.wsList[index_max] = (self.wsList[index_max] * self.wkList[index_max] + s) / (self.wkList[index_max] + 1)
        self.wNList[index_max] = self.wNList[index_max] + 1
        k = self.wkList[index_max]

        if self.histLen == self.data.shape[0]:
            print('the times of k:', self.wNList)
        # print(k)
        return k


# if __name__ == '__main__':
#
#     datasets = ['djia']
#     for d in datasets:
#         data = tools.dataset(d)
#         result = tools.quickrun(AT_ucb('djia', epsilon=50), data=data)
#         res = ListResult([result], ['AT_ucb'])
#         path = '/home/tjh/work/Fintech/git7-28/portfolio-risk/UPalgoTest/weightSave/'


if __name__ == '__main__':

    # datasetList = ['djia', 'tse', 'msci', 'nyse_n', 'hs300', 'EuroStoxx50', 'nasdaq(Medium+)']
    datasetList = [
        'NYSE_O',
        'NYSE_N',
        'NASDAQ',
        'tse',
        'hk29',
        # 'S&P500 Stock',  # 150 Correlation Prediction with ARIMA-LSTM Hybrid Model
        # 'csi300',
        # 'EuroStoxx50',
        'hs300',
        # 'russel1000'
    ]

    param_sensitive = {}

    for datasetName in datasetList:
        # data_path = CURRENT_PROJ_PATH + 'universal/data/' + datasetName + '.pkl'
        data_path = 'C:\\Users\\cgy\\Desktop\\UPalgoTest\\UPalgoTest\\universal\\data\\' + datasetName + '.pkl'
        df_original = pd.read_pickle(data_path)  # DataFrame

        # t = tools.quickrun(PPT(), df_original)
        t = AT_ucb(datasetName, epsilon=50)
        df = t._convert_prices(df_original, 'absolute')
        B = t.weights(df)
        B.to_csv('C:\\Users\\cgy\Desktop\\UPalgoTest\\UPalgoTest\\universal\\algo_weight\\STM_' + datasetName + '.csv', index=False)
