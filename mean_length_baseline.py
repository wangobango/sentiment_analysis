import numpy as np
import pandas as pd
import sys
import pickle
import os
import sys
import string
from config import Config
from data_evaluator import Evaluator
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

  def checkDataCorectness(self, setType = 'data'):
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
    
  def countCapitalLetters(self, string):
        return sum(1 for c in string if c.isupper())

  def teachLinearRegressionModel(self):
    self.lr = SGDClassifier(verbose=1)
    tokenizer = RegexpTokenizer(r'\w+')
    count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
    
    lenData = np.array([len(tokenizer.tokenize(x)) for idx, x in enumerate(self.dataSet['text']) if idx not in self.corruptedData['data']])
    capitalLettersData = np.array([self.countCapitalLetters(x) for idx, x in enumerate(self.dataSet['text']) if idx not in self.corruptedData['data']])
    punctuationMarksData = np.array([count(x, string.punctuation) for idx, x in enumerate(self.dataSet['text']) if idx not in self.corruptedData['data'] ])
    
    features = np.array([ lenData ])
    yData = np.array([[1 if x == 'positive' else 0 for idx, x in enumerate(self.dataSet['polarity']) if idx not in self.corruptedData['data']]])
    size = yData.size
    features = np.reshape(features, (size, 1))
    yData = np.reshape(yData, (size,))
    self.lr.fit(features, yData)

  def serializeModel(self):
    path = self.config.readValue("regression_model")
    dump(self.lr, path)

  def loadModel(self):
    path = self.config.readValue("regression_model")
    self.lr = load(path)

  def readTestSet(self):
    path = self.config.readValue('test_set_path')
    self.testSet = pd.read_csv(path)

  def evaluateModel(self):
    tokenizer = RegexpTokenizer(r'\w+')
    count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
    
    lenData = np.array([len(tokenizer.tokenize(x)) for idx, x in enumerate(self.testSet['text']) if idx not in self.corruptedData['test']])
    capitalLettersData = np.array([self.countCapitalLetters(x) for idx, x in enumerate(self.testSet['text']) if idx not in self.corruptedData['test']])
    punctuationMarksData = np.array([count(x, string.punctuation) for idx, x in enumerate(self.testSet['text']) if idx not in self.corruptedData['test'] ])
    
    features = np.array([ lenData ])
    correctPredictions = np.array([[1 if x == 'positive' else 0 for idx, x in enumerate(self.testSet['polarity']) if idx not in self.corruptedData['test']]])  
    
    features = np.reshape(features, (correctPredictions.size, 1))
    correctPredictions = np.reshape(correctPredictions, (correctPredictions.size, 1))
    
    self.predictions = self.lr.predict(features)
    print("\nEvaluation results:")
    self.evaluator.evaluate(correctPredictions, self.predictions)


if __name__ == "__main__":
  baseLine = MeanLengthBaseLine()
  if("-teach" in sys.argv):
    baseLine.readDataSet()
    baseLine.checkDataCorectness()
    baseLine.teachLinearRegressionModel()    
    baseLine.serializeModel()
  if("-evaluate" in sys.argv):
    baseLine.readTestSet()
    baseLine.checkDataCorectness('test')
    baseLine.loadModel()
    baseLine.evaluateModel()
