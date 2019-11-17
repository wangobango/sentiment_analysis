from .data_loader import Loader
from .data_loader import PolarityParser
from .config import Config
from pprint import pprint
from nltk.tokenize import RegexpTokenizer
from console_progressbar import ProgressBar
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import code
import xml.etree.ElementTree as ET
import numpy as np
import json
import sys
import string

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
        self.globalData = {}

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
            if(len(os.listdir("aggregated/")) != 0):
                os.system("rm -r aggregated")
                os.system("mkdir aggregated")
                return False

    def parsePolarityValue(self, value):
        return 1 if value == 'positive' else 0

    def parseData(self):
        loader = Loader()
        loader.set_parser(PolarityParser())
        self.config.addProperty(PROP,VALUE)
        topics = {}
        path = self.config.readValue(PATH)
        domains = os.listdir(path)
        self.domains = domains
        pb = ProgressBar(total=int(len(self.domains)-1),prefix='Data parsing in progress', suffix='', decimals=3, length=50, fill='X', zfill='-')
        frames = {}
        data = []

        for idx, topic in enumerate(domains):
            topics[topic] = []
            for item in os.listdir(path+topic):
                realPath = path + topic + "/" + item
                print(realPath)
                loader.set_path(realPath)
                # try:
                data = loader.repair_file().load()
                # except ET.ParseError as err:
                #     if '-debug' in sys.argv:
                #         print(err)
                #     loader.repair_encoding()

                if (len(data) > 0):
                    for sentance in data:
                        phrase = Phrase(*sentance.toArray())
                        topics[topic].append(phrase.toDictionary())
                else:
                    raise Exception('data length is 0')

            frames[topic] = pd.DataFrame(topics[topic])
            frames[topic].to_csv('aggregated/'+topic+'.csv')
            pb.print_progress_bar(idx)
                # print("Done topic: {}, {} / {}".format(topic, idx, len(domains)))
            self.frames = frames

    def readData(self):
        frames = {}
        path = self.config.readValue(PATH)
        self.domains = os.listdir(path)
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
            results[value] = 'dupa'
            if '-debug' in sys.argv:
                print(err)

    def countCapitalLetters(self, string):
        return sum(1 for c in string if c.isupper())

    def analyzeAllDomains(self):
        pb = ProgressBar(total=int(len(self.domains)-1),prefix='Domain analysis in progress', suffix='', decimals=3, length=50, fill='X', zfill='-')
        data = self.getFrames()
        for idx,domain in enumerate(self.getDomains()):
            topic = data[domain]
            arr = topic.to_numpy()
            self.analyzeByDomain(domain, arr)
            self.calculateGlobalData(arr)
            pb.print_progress_bar(idx)

    def calculateGlobalData(self, arr):
        # 1# Get data for global punctuation marks count
        pass

    def analyzeByDomain(self,domain, arr):
        """
            Data summary :
                arr - array of data aggragated for a specific domain
                domain - topic domain for which data is being aggregated

                    arr columns consist of :
                        * 0 - id generated by pandas
                        * 1 - domain
                        * 2 - id provided by author of the data set
                        * 3 - polarity
                        * 4 - summary 
                        * 5 - text
        """
        ID = 0
        DOMAIN = 1
        GIVEN_ID = 2
        POLARITY = 3
        SUMMARY = 4
        TEXT = 5

        results = {}
        tokenizer = RegexpTokenizer(r'\w+')
        count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))

        print(arr)

        # self.setResultsValue(results, 'domain', domain, lambda x, y: x)
        self.setResultsValue(results, 'numberOfPositives', arr, lambda arr, results: (arr[:,3] == 'positive').sum())
        self.setResultsValue(results, 'numberOfNegatives', arr, lambda arr, results: len(arr[:,0]) - results['numberOfPositives'])
        self.setResultsValue(results, 'positiveToNegativeRatio', arr, lambda arr, results: results['numberOfPositives']/results['numberOfNegatives'] if results['numberOfNegatives'] != 0 else results['numberOfPositives'])
        self.setResultsValue(results, 'meanTextLengthCharacters', arr, lambda arr, results: np.sum([len(x) for x in arr[:,5]])/len(arr[:,5]))
        self.setResultsValue(results, 'meanTextLengthWords', arr, lambda arr, results: np.sum([len(tokenizer.tokenize(x)) for x in arr[:,5]])/len(arr[:,5]))
        self.setResultsValue(results, 'averageTextLengthWhenPolarityPositiveChars', arr, lambda arr, results: np.sum([len(x[2]) for x in arr[:,3:6] if x[0] == 'positive']) / results['numberOfPositives'])
        self.setResultsValue(results, 'averageTextLengthWhenPolarityPositiveWords', arr, lambda arr, results: np.sum([len(tokenizer.tokenize(x[2])) for x in arr[:,3:6] if x[0] == 'positive']) / results['numberOfPositives'])
        self.setResultsValue(results, 'averageTextLengthWhenPolarityNegativeChars', arr, lambda arr, results: np.sum([len(x[2]) for x in arr[:,3:6] if x[0] == 'negative']) / results['numberOfNegatives'] )
        self.setResultsValue(results, 'averageTextLengthWhenPolarityNegativeWords', arr, lambda arr, results: np.sum([len(tokenizer.tokenize(x[2])) for x in arr[:,3:6] if x[0] == 'negative']) / results['numberOfNegatives'])
        self.setResultsValue(results, 'averageTextLengthWhenPolarityNegativeWords', arr, lambda arr, results: np.sum([len(tokenizer.tokenize(x[2])) for x in arr[:,3:6] if x[0] == 'negative']) / results['numberOfNegatives'])
        self.setResultsValue(results, 'averageNumberOfCapitalLettersPolarityPositive', arr, lambda arr, results: np.sum([self.countCapitalLetters(x[2]) for x in arr[:,3:6] if x[0] == 'positive'])/ results['numberOfPositives'])
        self.setResultsValue(results, 'averageNumberOfCapitalLettersPolarityNegative', arr, lambda arr, results: np.sum([self.countCapitalLetters(x[2]) for x in arr[:,3:6] if x[0] == 'negative'])/ results['numberOfNegatives'])
        self.setResultsValue(results, 'stdDevTextLengthWhenPolarityPositive', arr, lambda arr, results: np.std([len(x[2]) for x in arr[:,3:6] if x[0] == 'positive']))
        self.setResultsValue(results, 'stdDevTextLengthWhenpolarityNegative', arr, lambda arr, resutls: np.std([len(x[2]) for x in arr[:,3:6] if x[0] == 'negative']))
        self.setResultsValue(results, 'stdDevNumOfCapitalLettersPolarityPositive', arr, lambda arr, results: np.std([self.countCapitalLetters(x[2]) for x in arr[:,3:6] if x[0] == 'positive']))
        self.setResultsValue(results, 'stdDevNumOfCapitalLettersPolarityNegative', arr, lambda arr, results: np.std([self.countCapitalLetters(x[2]) for x in arr[:,3:6] if x[0] == 'negative']))
        self.setResultsValue(results, 'covBetweenTextLengthAndNumOfCapitalLetters', arr, lambda arr, resulsts: np.cov(([len(tokenizer.tokenize(x[2])) for x in arr[:,3:6]], [self.countCapitalLetters(x[2]) for x in arr[:,3:6]])))
        self.setResultsValue(results, 'covBetweenTextLengthAndPolarity', arr, lambda arr, results: np.cov(([len(tokenizer.tokenize(x[2])) for x in arr[:,3:6]], [1 if x[0]=='positive' else 0 for x in arr[:,3:6]])))
        self.setResultsValue(results, 'personCorBetweenTextLengthAndNumOfCapitalLetters', arr, lambda arr, results: np.corrcoef(([len(tokenizer.tokenize(x[2])) for x in arr[:,3:6]], [1 if x[0]=='positive' else 0 for x in arr[:,3:6]])))
        self.setResultsValue(results, 'personCorBetweenTextLengthAndPolarity', arr, lambda arr, results: np.corrcoef(([len(tokenizer.tokenize(x[2])) for x in arr[:,3:6]], [1 if x[0]=='positive' else 0 for x in arr[:,3:6]])))
        self.setResultsValue(results, 'meanPunctuationMarksPolarityPositive', arr, lambda arr, results: np.sum([count(x[2], string.punctuation) for x in arr[:,3:6] if x[0] == 'positive']) / results['numberOfPositives'] )
        self.setResultsValue(results, 'meanPunctuationMarksPolarityNegative', arr, lambda arr, results: np.sum([count(x[2], string.punctuation) for x in arr[:,3:6] if x[0] == 'negative']) / results['numberOfNegatives'] )

        self.analyzedDataByDomain[domain] = results


    def dumpResultsToJSON(self):
        path = self.config.readValue('results_path')
        with open(path, "w") as f:
            json.dump(self.analyzedDataByDomain, f, cls=NpEncoder)
        
    """
    TODO
    Needs fix - at this point raises JSON.DecodeError
    """
    def loadResultsFromJson(self):
        path = self.config.readValue('results_path')
        return json.loads(path)

    def preparePlotPresentation(self):
            pp = self.PlotPresentation(self)
            pp.plotTextLengthByDomain()
            pp.plotPunctutionMarksByDomain()

    class PlotPresentation:
        def __init__(self, parent):
            self.parent = parent
            self.plotPath = self.parent.config.readValue('plot_path')
            # matplotlib.use('Agg')

        def autolabel(self, rects, ax):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        def plotTextLengthByDomain(self):
            labels = self.parent.domains
            negativeMeans = [int(self.parent.analyzedDataByDomain[x]['averageTextLengthWhenPolarityNegativeWords']) for x in labels]
            positiveMeans = [int(self.parent.analyzedDataByDomain[x]['averageTextLengthWhenPolarityPositiveWords']) for x in labels]
            x = np.arange(len(labels))
            width = 0.35

            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width/2, negativeMeans, width, label='Mean Positive Length')
            rects2 = ax.bar(x + width/2, positiveMeans, width, label='Mean Negative Length')

            ax.set_ylabel('Domains')
            ax.set_title('Mean positive and negative text length by domain')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()

            self.autolabel(rects1, ax)
            self.autolabel(rects2, ax)

            fig.tight_layout()
            plt.savefig(self.plotPath+'mean_positive_negative_by_domain.png')
            plt.show()

        def plotPunctutionMarksByDomain(self):
            labels = self.parent.domains
            positiveMarks = [int(self.parent.analyzedDataByDomain[x]['meanPunctuationMarksPolarityPositive']) for x in labels]
            negativeMarks = [int(self.parent.analyzedDataByDomain[x]['meanPunctuationMarksPolarityNegative']) for x in labels]

            x = np.arange(len(labels))
            width = 0.35

            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width/2, positiveMarks, width, label='Mean Positive Length')
            rects2 = ax.bar(x + width/2, negativeMarks, width, label='Mean Negative Length')

            ax.set_ylabel('Domains')
            ax.set_title('Mean positive and negative count of punctuation marks by domain')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()

            self.autolabel(rects1, ax)
            self.autolabel(rects2, ax)

            fig.tight_layout()
            plt.savefig(self.plotPath+'mean_punctuation_by_domain.png')
            plt.show()


        

if __name__ == "__main__":
    """
        USAGE:
            - Run with command: 'python3 data_exploration'
            - Add flag -plot to draw plots
            - Add flag -dump to dump results file to json
            - Add flag -debug to print error logs
            - Add flag -aggregate to aggregate data in aggregated directory
    """    
    
    de = DataExplorer()
    if("-aggregate" in sys.argv):
        de.start()
        de.parseData()
    else:
        de.readData()

    # Initial analysis
    de.analyzeAllDomains()

    if("-dump" in sys.argv):
        de.dumpResultsToJSON()

    if("-plot" in sys.argv):
        de.preparePlotPresentation()

