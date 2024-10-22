from sklearn import metrics
import numpy as np
import pandas as pd

class Evaluator:
    # 0,0 TN
    # 1,0 FN
    # 1,1 TP
    # 0,1 FP
    FSCORE = 'f-score'
    PRECISION = 'precision'
    RECALL = 'recall'
    ACCURACY = 'accuracy'
    C_MATRIX = 'confusion-matrix'

    def evaluate(self, expectedResults, actualResults, printConfusionMatrix = False):
        evaluatedMetircs = {}
        confusionMatrix = self.calculate_confusion_matrix(expectedResults, actualResults)
        # evaluatedMetircs[self.C_MATRIX] = confusionMatrix.flatten()
        evaluatedMetircs[self.ACCURACY] = self.calculate_accuracy()
        evaluatedMetircs[self.PRECISION] = self.calculate_precision()
        evaluatedMetircs[self.RECALL] = self.calculate_recall()
        evaluatedMetircs[self.FSCORE] = self.calculate_fscore(evaluatedMetircs[self.PRECISION], evaluatedMetircs[self.RECALL])
        self.print(evaluatedMetircs)
        self.evaluatedMetircs = evaluatedMetircs
        return evaluatedMetircs

    def getRecall(self):
        return self.evaluatedMetircs[self.RECALL]

    def getPrecision(self):
        return self.evaluatedMetircs[self.PRECISION]

    def getAccuracy(self):
        return self.evaluatedMetircs[self.ACCURACY]

    def getFScore(self):
        return self.evaluatedMetircs[self.FSCORE]    

    def calculate_precision(self):
        return self.tp / (self.tp + self.fp)

    def calculate_recall(self):
        return self.tp / (self.tp + self.fn)

    def calculate_fscore(self, precsision, recall):
        return 2*precsision*recall / (precsision+recall)

    def calculate_accuracy(self):
        return (self.tp + self.tn)/(self.tp + self.tn + self.fn + self.fp)
    
    def calculate_confusion_matrix(self, expectedResults, actualResults):
        confusionMatrix = metrics.confusion_matrix(expectedResults, actualResults)
        self.tp = confusionMatrix[1][1]
        self.tn = confusionMatrix[0][0]
        self.fn = confusionMatrix[1][0]
        self.fp = confusionMatrix[0][1]
        self.confusionMatrix = confusionMatrix
        return confusionMatrix

    def getString(self, evaluatedMetrics): 
        text = ""
        for i, item in enumerate(evaluatedMetrics.items()):
            if i > 0:
                text += "\t" if i % 2 == 0 else "\n"
            text += str(item[0]) + ": " + str(item[1])
        return text

    def print(self, evaluatedMetrics):
        print(self.getString(evaluatedMetrics))
