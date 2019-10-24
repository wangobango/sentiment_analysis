from data_loader import DataLoader
from config import Config
from pprint import pprint
from nltk.tokenize import RegexpTokenizer
import os
import pandas as pd
import code
import xml.etree.ElementTree as ET
import numpy as np
import json

PROP = "CURRENT_DATA"
VALUE = "./data/Amazon_Instant_Video/Amazon_Instant_Video.neg.0.xml"
PATH = "data_path"
COLUMNS = ["id", "domain", "polarity", "summmary", "text"]

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class Phrase():
    def __init__(self, id, domain, polarity, summary, text):
        self.id = id
        self.domain = domain
        self.polarity = polarity
        self.summary = summary
        self.text = text

    def toString(self):
        return 'Id: {},\nDomain: {},\nPolarity: {},\nSummary: {},\nText: {} \n'.format(self.id, self.domain, self.polarity, self.summary, self.text)

    def toDictionary(self):
        return {"id": self.id, "domain": self.domain, "polarity": self.polarity, "summary": self.summary, "text": self.text}

class DataExplorer():
    def __init__(self):
        self.config = Config()
        self.domains = []
        self.frames = {}
        self.analyzedDataByDomain = {}

    def start(self):
        items = os.listdir('./')
        if not 'aggregated' in items:
            os.system('mkdir aggregated')
            return True
        else:
            return False

    def parsePolarityValue(self, value):
        return 1 if value == 'positive' else 0

    def parseData(self):
        loader = DataLoader()
        self.config.addProperty(PROP,VALUE)
        topics = {}
        path = self.config.readValue(PATH)
        domains = os.listdir(path)
        self.domains = domains
        frames = {}

        for topic in domains:
            topics[topic] = []
            for item in os.listdir(path+topic):
                realPath = path + topic + "/" + item
                loader.set_path(realPath)
                try:
                    data = loader.read_xml()
                except ET.ParseError as err:
                    print(err)
                    loader.repair_file(err.position[0], err.position[1])

                for sentance in data:
                    phrase = Phrase(*sentance.toArray())
                    topics[topic].append(phrase.toDictionary())

                frames[topic] = pd.DataFrame(topics[topic])
                frames[topic].to_csv('aggregated/'+topic+'.csv')

            self.frames = frames

    def readData(self):
        frames = {}
        path = self.config.readValue(PATH)
        self.domains = os.listdir(path)
        counter = 0
        for topic in self.domains:
            frames[topic] = pd.read_csv('aggregated/'+topic+'.csv', delimiter=',')
           
        self.frames = frames

    def getFrames(self):
        return self.frames

    def getDomains(self):
        return self.domains

    def setResultsValue(self, results, value, arr, function):
        try:
            results[value] = function(arr, results)
        except TypeError as err:
            results[value] = 'error? but whyyyy'
            print(err)

    def analyzeByDomain(self, domain):
        data = self.getFrames()
        topic = data[domain]
        arr = topic.to_numpy()

        results = {}
        tokenizer = RegexpTokenizer(r'\w+')

        self.setResultsValue(results, 'domain', 'domain', lambda x, y: x)
        self.setResultsValue(results, 'numberOfPositives', arr, lambda arr, results: (arr[:,3] == 'positive').sum())
        self.setResultsValue(results, 'numberOfNegatives', arr, lambda arr, results: len(arr[:,0]) - results['numberOfPositives'])
        self.setResultsValue(results, 'positiveToNegativeRatio', arr, lambda arr, results: results['numberOfPositives']/results['numberOfNegatives'] if results['numberOfNegatives'] != 0 else results['numberOfPositives'])
        self.setResultsValue(results, 'meanTextLengthCharacters', arr, lambda arr, results: np.sum([len(x) for x in arr[:,5]])/len(arr[:,5]))
        self.setResultsValue(results, 'meanTextLengthWords', arr, lambda arr, results: np.sum([len(tokenizer.tokenize(x)) for x in arr[:,5]])/len(arr[:,5]))
        self.setResultsValue(results, 'averageTextLengthWhenPolarityPositiveChars', arr, lambda arr, results: np.sum([len(x[2]) for x in arr[:,3:6] if x[0] == 'positive']) / results['numberOfPositives'])
        self.setResultsValue(results, 'averageTextLengthWhenPolarityPositiveWords', arr, lambda arr, results: np.sum([len(tokenizer.tokenize(x[2])) for x in arr[:,3:6] if x[0] == 'positive']) / results['numberOfPositives'])
        self.setResultsValue(results, 'averageTextLengthWhenPolarityNegativeChars', arr, lambda arr, results: np.sum([len(x[2]) for x in arr[:,3:6] if x[0] == 'negative']) / results['numberOfNegatives'] )
        self.setResultsValue(results, 'averageTextLengthWhenPolarityNegativeChars', arr, lambda arr, results: np.sum([len(x[2]) for x in arr[:,3:6] if x[0] == 'negative']) / results['numberOfNegatives'])
        self.setResultsValue(results, 'averageTextLengthWhenPolarityNegativeWords', arr, lambda arr, results: np.sum([len(tokenizer.tokenize(x[2])) for x in arr[:,3:6] if x[0] == 'negative']) / results['numberOfNegatives'])

        self.analyzedDataByDomain[domain] = results

    def dumpResultsToJSON(self):
        path = self.config.readValue('results_path')
        with open(path, "a") as f:
            json.dump(self.analyzedDataByDomain, f, cls=NpEncoder)
        


if __name__ == "__main__":
    de = DataExplorer()
    if(de.start()):
        de.parseData()
    else:
        de.readData()

    # Initial analysis
    for domain in de.getDomains():
        de.analyzeByDomain(domain)

    de.dumpResultsToJSON()
    # TODO: Visualize results data
    # TODO: Corrlation analysis - are variables somehow correlated ?
    # TODO: Difference in results between different domains
