import numpy as np
import pandas as pd
import sys
import pickle
import os
import sys
import string
import pandas as pd
from utils.config import Config
from utils.data_evaluator import Evaluator
from sklearn.linear_model import SGDClassifier
from nltk.tokenize import RegexpTokenizer
from scipy.sparse import csr_matrix
from joblib import dump, load
from sklearn.svm import SVC
from itertools import chain, combinations



class MeanLengthBaseLine:
    def __init__(self):
        self.config = Config()
        self.evaluator = Evaluator()
        self.corruptedData = {}
        self.models = {}
        self.results = pd.DataFrame(columns = ["domain", "features", "n_features", "accuracy", "precision", "recall", "fscore"])

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
        # self.dataSet = self.dataSet[:10000]

    def countCapitalLetters(self, string):
        return sum(1 for c in string if c.isupper())

    def all_subsets(self, ss):
        return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

    def teachLinearModelWrapper(self):
        domains =  self.dataSet.groupby("domain").count()['polarity'].keys()
        features = ['lenData', 'capitals', 'punctuation']
        ss = self.all_subsets(features)
        for idx, f in enumerate(ss):
            if(idx == 0):
                continue
            feet = list(f)
            print(feet)
            if(len(feet) == 0 ):
                continue
            for domain in domains:
                print("Teaching for domain:{}, and features: {}".format(domain, feet))
                self.models[domain] = self.teachLinearRegressionModel(domain, feet)
            if("-all" in sys.argv):
                self.readTestSet()
                print("Evaluating")
                self.evaluateModelWrapper(feet)
                self.results.to_csv("dupa.csv", sep=";")

        self.results.to_csv("dupa.csv", sep=";")


    def teachLinearRegressionModel(self, domain, f):
        model = SGDClassifier(verbose=0)
        tokenizer = RegexpTokenizer(r'\w+')
        def count(l1, l2): return len(list(filter(lambda c: c in l2, l1)))

        dataSet = self.dataSet
        dataSet[dataSet.apply(lambda row: row['domain'] == domain, axis = 1)]

        features = []
        numOfFeatures = 0

        if("lenData" in f):
            lenData = np.array([len(tokenizer.tokenize(x)) for idx, x in enumerate(
                dataSet['text']) ])
            features.append(lenData)
            numOfFeatures += 1
        if("capitals" in f):
            capitalLettersData = np.array([self.countCapitalLetters(x) for idx, x in enumerate(
                dataSet['text']) ])
            features.append(capitalLettersData)
            numOfFeatures += 1
        if("punctuation" in f):
            punctuationMarksData = np.array([count(x, string.punctuation) for idx, x in enumerate(
                dataSet['text']) ])
            features.append(punctuationMarksData)
            numOfFeatures += 1

        features = np.asarray(features)

        yData = [1 if x == 'positive' else 0 for idx, x in enumerate(
            dataSet['polarity']) ]
        size = len(yData)
        features = np.reshape(features, (size, numOfFeatures))
        model.fit(features, yData)
        return model

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
        # self.testSet = self.testSet[:1000]

    def evaluateModelWrapper(self, features):
        correctPredictions_array = []
        predictions_array = []
        for domain, model in self.models.items():
            features, correctPredictions = self.evaluateModel(domain, model, features)
            # print('feat:')
            # print(features)
            # print('correct:')
            # print(correctPredictions)
            correctPredictions_array.extend(correctPredictions)
            pred = model.predict(features)
            predictions_array.extend(pred)
            # print('predicted')
            # print(pred)
            # self.evaluator.evaluate(correctPredictions_array, predictions_array)
            self.evaluator.evaluate(correctPredictions, pred)
            self.results = self.results.append({'domain': domain,'features': features, "n_features": len(features), "accuracy": self.evaluator.getAccuracy(), "precision": self.evaluator.getPrecision(), "recall": self.evaluator.getRecall(), "fscore": self.evaluator.getFScore() }, ignore_index = True)
        print("\nEvaluation results:")
        self.evaluator.evaluate(correctPredictions_array, predictions_array)
        self.results = self.results.append({'domain':'all','features': features, "n_features": len(features), "accuracy": self.evaluator.getAccuracy(), "precision": self.evaluator.getPrecision(), "recall": self.evaluator.getRecall(), "fscore": self.evaluator.getFScore() }, ignore_index = True)



    def evaluateModel(self, domain, model, f):
        print('evaluating domain: ' + domain)
        print(f)

        tokenizer = RegexpTokenizer(r'\w+')
        def count(l1, l2): return len(list(filter(lambda c: c in l2, l1)))

        testSet = self.testSet
        testSet[testSet.apply(lambda row: row['domain'] == domain, axis = 1)]
        testSet = testSet.sample(frac=1).reset_index(drop=True)

        features = []
        numOfFeatures = 0

        if("lenData" in f):
            lenData = np.array([len(tokenizer.tokenize(x)) for idx, x in enumerate(
                testSet['text']) ])
            features.append(lenData)
            numOfFeatures += 1
        if("capitals" in f):
            capitalLettersData = np.array([self.countCapitalLetters(x) for idx, x in enumerate(
                testSet['text']) ])
            features.append(capitalLettersData)
            numOfFeatures += 1
        if("punctuation" in f):
            punctuationMarksData = np.array([count(x, string.punctuation) for idx, x in enumerate(
                testSet['text']) ])
            features.append(punctuationMarksData)
            numOfFeatures += 1

        features = np.asarray(features)


        correctPredictions = [1 if x == 'positive' else 0 for idx, x in enumerate(
            testSet['polarity'])]

        features = np.reshape(features, (len(correctPredictions), numOfFeatures))
        # correctPredictions = np.reshape(
        #     correctPredictions, (correctPredictions.size, 1))

        return features, correctPredictions


if __name__ == "__main__":
    """
        -all to combine stuff
    """
    baseLine = MeanLengthBaseLine()
    if("-teach" in sys.argv):
        baseLine.readDataSet()
        baseLine.teachLinearModelWrapper()
        baseLine.serializeModel()
    if("-evaluate" in sys.argv):
        baseLine.readTestSet()
        baseLine.loadModel()
        baseLine.evaluateModelWrapper("")
