import numpy as np
import pandas as pd
import sys
import pickle
import os
import sys
import string
from utils.config import Config
from utils.data_evaluator import Evaluator
from sklearn.linear_model import SGDClassifier
from nltk.tokenize import RegexpTokenizer
from scipy.sparse import csr_matrix
from joblib import dump, load
from sklearn.svm import SVC


class MeanLengthBaseLine:
    def __init__(self):
        self.config = Config()
        self.evaluator = Evaluator()
        self.corruptedData = {}
        self.models = {}

    def checkDataCorectness(self, setType='data'):
        tokenizer = RegexpTokenizer(r'\w+')

        if setType == 'data':
            dataSetToCheck = self.dataSet['text']
            self.corruptedData['data'] = []
        else:
            dataSetToCheck = self.testSet['text']
            self.corruptedData['test'] = []

        for idx, item in enumerate(dataSetToCheck):
            try:
                test = len(tokenizer.tokenize(item))
            except TypeError as idf:
                print(idf)
                self.corruptedData[setType].append(idx)
                print("Error at id: {}".format(idx))
                print(test)
                print(item)

    def readDataSet(self):
        path = self.config.readValue('data_set_path')
        self.dataSet = pd.read_csv(path)
        self.dataSet.dropna(inplace=True)

    def countCapitalLetters(self, string):
        return sum(1 for c in string if c.isupper())

    def teachLinearModelWrapper(self):
        domains =  self.dataSet.groupby("domain").count()['polarity'].keys()

        for domain in domains:
            self.models[domain] = self.teachLinearRegressionModel(domain)


    def teachLinearRegressionModel(self, domain):
        self.lr = SGDClassifier(verbose=1)
        tokenizer = RegexpTokenizer(r'\w+')
        def count(l1, l2): return len(list(filter(lambda c: c in l2, l1)))

        dataSet = self.dataSet
        dataSet[dataSet.apply(lambda row: row['domain'] == domain, axis = 1)]

        lenData = np.array([len(tokenizer.tokenize(x)) for idx, x in enumerate(
            dataSet['text']) if idx not in self.corruptedData['data']])
        capitalLettersData = np.array([self.countCapitalLetters(x) for idx, x in enumerate(
            dataSet['text']) if idx not in self.corruptedData['data']])
        punctuationMarksData = np.array([count(x, string.punctuation) for idx, x in enumerate(
            dataSet['text']) if idx not in self.corruptedData['data']])

        features = np.array([lenData])
        yData = np.array([[1 if x == 'positive' else 0 for idx, x in enumerate(
            dataSet['polarity']) if idx not in self.corruptedData['data']]])
        size = yData.size
        features = np.reshape(features, (size, 1))
        yData = np.reshape(yData, (size,))
        self.lr.fit(features, yData)
        return self.lr

    def serializeModel(self):
        path = self.config.readValue("regression_model")
        dump(self.models, path)

    def loadModel(self):
        path = self.config.readValue("regression_model")
        self.models = load(path)

    def readTestSet(self):
        path = self.config.readValue('test_set_path')
        self.testSet = pd.read_csv(path)
        self.testSet.dropna(inplace=True)

    def evaluateModelWrapper(self):
        correctPredictions_array = []
        predictions_array = []
        for domain, model in self.models.items():
            features, correctPredictions = self.evaluateModel(domain, model)
            print('feat:')
            print(features)
            print('correct:')
            print(correctPredictions)
            correctPredictions_array.extend(correctPredictions)
            pred = model.predict(features)
            predictions_array.extend(pred)
            print('predicted')
            print(pred)
            self.evaluator.evaluate(correctPredictions_array, predictions_array)
        print("\nEvaluation results:")
        self.evaluator.evaluate(correctPredictions_array, predictions_array)



    def evaluateModel(self, domain, model):
        print('evaluating domain: ' + domain)
        tokenizer = RegexpTokenizer(r'\w+')
        def count(l1, l2): return len(list(filter(lambda c: c in l2, l1)))

        testSet = self.testSet
        testSet[testSet.apply(lambda row: row['domain'] == domain, axis = 1)]
        testSet = testSet.sample(frac=1).reset_index(drop=True)
        testSet = testSet[:100]
        lenData = np.array([len(tokenizer.tokenize(x)) for idx, x in enumerate(testSet['text'])])
        # capitalLettersData = np.array([self.countCapitalLetters(x) for idx, x in enumerate(
        #     testSet['text']) if idx not in self.corruptedData['test']])
        # punctuationMarksData = np.array([count(x, string.punctuation) for idx, x in enumerate(
        #     testSet['text']) if idx not in self.corruptedData['test']])

        features = lenData

        correctPredictions = np.array([1 if x == 'positive' else 0 for idx, x in enumerate(
            testSet['polarity']) if idx not in self.corruptedData['test']])

        features = np.reshape(features, (correctPredictions.size, 1))
        correctPredictions = np.reshape(
            correctPredictions, (correctPredictions.size, 1))

        return features, correctPredictions

        # self.predictions = self.lr.predict(features)
        # print("\nEvaluation results:")
        # self.evaluator.evaluate(correctPredictions, self.predictions)


if __name__ == "__main__":
    baseLine = MeanLengthBaseLine()
    if("-teach" in sys.argv):
        baseLine.readDataSet()
        baseLine.checkDataCorectness()
        baseLine.teachLinearModelWrapper()
        baseLine.serializeModel()
    if("-evaluate" in sys.argv):
        baseLine.readTestSet()
        baseLine.checkDataCorectness('test')
        baseLine.loadModel()
        baseLine.evaluateModelWrapper()
