import json
import sys
import os
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from .config import Config
from pprint import pprint
from .data_loader import Loader
from nltk.tokenize import RegexpTokenizer

PATH = "data_path"

class ResultsProcessor:

    def __init__(self):
        self.config = Config()

    def getDomainNames(self):
        return os.listdir("data/")

    def loadResults(self):
        path = self.config.readValue("results_path")
        with (open(path, "r")) as f:
            self.results = json.load(f)
    
    def getValueFromAllDomains(self, value):
        results = {}
        for (key, val) in self.results.items():
            results[key] = val[value]
        pprint(results)
        return results
    
    def readData(self):
        frames = {}
        path = self.config.readValue(PATH)
        self.domains = os.listdir(path)
        for topic in self.domains:
            frames[topic] = pd.read_csv('aggregated/'+topic+'.csv', delimiter=',')
           
        self.frames = frames

    """
        Creates `word : number of times this word occures` dictionary.
    """
    def calculateWordFreqs(self):
        print("Reading data")
        self.readData()
        occurances = {}
        tokenizer = RegexpTokenizer(r'\w+')
        print("Creating `word:number of occurances` dictionary")
        for key, domain in self.frames.items():
            # print(domain)
            for idx, row in domain.iterrows():
                if(isinstance(row["text"], str)):
                    for word in tokenizer.tokenize(row["text"]):
                        if(word in occurances):
                            occurances[word] += 1
                        else:
                            occurances[word] = 1
                else:
                    print("Corrupted text at idx: {}, and domain: {} \n".format(idx,key))
        print("Finished parsing")
        self.occurances = occurances

    def occuranceHelper(self, values):
        pass

    def calculateWordOccurances(self):
        parsed_dict = {}
        print("Parsing occurances dictionary to `occurance number : list of words` dictionary")
        for key,value in self.occurances.items():
            if(value in parsed_dict):
                parsed_dict[value].append(key)
            else:
                parsed_dict[value] = [key]
        print("Finished parsing")
        self.parsed_dict = parsed_dict

    def dumpOccurancesToJson(self):
        new_occurances = {}
        for key,value in self.parsed_dict.items():
            new_occurances[key] = len(value)

        sorted_dict = [(i, new_occurances[i]) for i in sorted(new_occurances.keys())]

        with(open("occurances.json", "w")) as f:
            json.dump(sorted_dict, f)

    def dumpParsedDictToJson(self):
        with(open("parsed.json", "w")) as f:
            json.dump(self.parsed_dict, f)
        

    def serializeDict(self):
        with(open("dict.pickl", "wb")) as f:
            pickle.dump(self.parsed_dict, f)

    def loadDict(self):
        with(open("dict.pickl", "rb")) as f:
            self.parsed_dict = pickle.load(f)

    def autolabel(self, rects, ax):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

    def createPlot(self):
        labels = range(1,5,1)
        data = []
        for label in labels:
            data.append(len(self.parsed_dict[label]))

        x = np.arange(len(labels))
        width = 0.1
        fig, ax = plt.subplots()
        rects = []
        for idx,item in enumerate(data):
            rects.append(ax.bar(x - int(width/(idx+1)), item, width, label = str(idx)))

        ax.set_ylabel('Word frequencues')
        ax.set_title('Word frequencies')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        for rect in rects:
            self.autolabel(rect, ax)
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    rp = ResultsProcessor()
    if("-res" in sys.argv):
        rp.loadResults()
        rp.getValueFromAllDomains(sys.argv[2])
    elif("-freq" in sys.argv):
        rp.calculateWordFreqs()
        rp.calculateWordOccurances()
        rp.serializeDict()
    elif("-plot" in sys.argv):
        rp.loadDict()
        # print(rp.parsed_dict[1])
        # rp.createPlot()
        rp.dumpOccurancesToJson()
        rp.dumpParsedDictToJson()