import csv
import os
import pandas as pd
import heapq
from universal import tools
from universal import algos
from universal.algo import Algo

import random, datetime
import logging
from MyLogger import MyLogger

import numpy as np

# we would like to see algos progress
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

import matplotlib
import matplotlib.pyplot as plt
from MultiShower import MultiShower
from SimpleSaver import SimpleSaver


class Tester:

    def __init__(self):
        self.data = None
        self.algo = None
        self.result = None
        self.X = None
        # self.logger = MyLogger('PTester_summary')
        self.saver = SimpleSaver()
        self.datasetName = None
        self.NStocks = 0

    def createDataSet(self, datasetName):
        # load data using tools module
        self.data = tools.dataset(datasetName)
        # self.data = self.data.iloc[:500]
        self.datasetName = datasetName
        print('data.type: ', type(self.data))
        self.NStocks = self.data.shape[1]
        print(self.data.head())
        print(self.data.shape)

    def slimDataSet(self, numCols=5):
        # invoked after createDataSet
        n, m = self.data.shape
        # random.randint(1, 10)  # Integer from 1 to 10, endpoints included
        sels = []
        df = pd.DataFrame()
        labels = self.data.columns
        while len(sels) < numCols:
            j = random.randint(0, m - 1)
            if j in sels:
                continue
            df[labels[j]] = self.data.iloc[:, j]
            sels.append(j)

        self.data = df
        print('slim_' + self.datasetName + '_', self.data)
        self.NStocks = self.data.shape[1]

    def createRatioX(self):
        PRICE_TYPE = 'ratio'
        self.data = tools.dataset('nyse_o')
        X = Algo._convert_prices(self.data, PRICE_TYPE)
        print('X: ', X)
        mX = X.to_numpy()
        print('shape: ', mX.shape, 'mX:', mX)

    def showNpArray(self):
        arr = self.data.to_numpy()
        print('df.shape: ', self.data.shape, 'arr.shape: ', arr.shape)  # (5651, 36)

    def showNRows(self, index, window):
        rows, cols = self.data.shape
        start = index
        if index > rows:
            start = rows - window
        if index < 0:
            index = 0
        end = start + window
        if end >= cols:
            end = cols - 1
        df = self.data.iloc[range(start, end)]

        print('[ ' + str(start) + ',' + str(end) + ')', df)

    def showdfIndex(self):
        ind = self.data.index
        print('df.index: ', ind)

    def getDataSetNameWithDT(self):
        tMark = str(datetime.datetime.now())
        return self.datasetName

    def showResult(self, d):
        from universal.algos.AT_ucb import AT_ucb

        ms = MultiShower(self.datasetName)
        for fee in [0]:
            result_STM = AT_ucb(d, 50).run(self.data)

            result_STM.fee = fee

            ms.show(
                [
                    result_STM,
                ],
                [
                    'STM',
                ],
                yLable=self.datasetName.upper() + '  Cumulative Wealth'
            )
            plt.show()


    @staticmethod
    def testSimple():
        datasets = ['nyse_n', 'nyse_o', 'tse', 'nasdaq', 'hs300', 'hk29']
        for d in datasets:
            t = Tester()
            t.createDataSet(d)
            t.showResult(d)




if __name__ == '__main__':
    Tester.testSimple()
