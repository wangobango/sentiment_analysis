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
from scipy import stats
import collections
from statsmodels.formula.api import ols
from Levenshtein import jaro_winkler
from collections import Counter

PATH = "data_path"
TOKENIZER = RegexpTokenizer(r'\w+')
FUNCTION_DEFINITIONS = {
    "mean_word_length": lambda data_set : sum([len(TOKENIZER.tokenize(x)) for x in data_set["text"] if isinstance(x,str)])/len(data_set["text"]),
    "std": lambda data_set : np.std(np.asarray([len(TOKENIZER.tokenize(x)) for x in data_set["text"] if isinstance(x,str)])),
    "median": lambda data_set : np.median(np.asarray([len(TOKENIZER.tokenize(x)) for x in data_set["text"] if isinstance(x,str)])),
    "variance": lambda data_set: np.var(np.asarray([len(TOKENIZER.tokenize(x)) for x in data_set["text"] if isinstance(x,str)])),
    "lenghts": lambda data_set : [len(TOKENIZER.tokenize(x)) for x in data_set["text"] if isinstance(x,str)],
    "polarity_positive_lenghts":  lambda data_set : [len(TOKENIZER.tokenize(x)) for x,y in zip(data_set["text"], data_set["polarity"]) if isinstance(x,str) and y == 'positive'],
    "polarity_negative_lenghts":  lambda data_set : [len(TOKENIZER.tokenize(x)) for x,y in zip(data_set["text"], data_set["polarity"]) if isinstance(x,str) and y == 'negative']
}

class Vocabulary:
    def __init__(self, vocab, vocab2int):
        self.vocabulary = vocab
        self.vocab2int = vocab2int
        self.config = Config()

    def getVocab(self):
        return self.vocabulary

    def getVocab2int(self):
        return self.vocab2int

    def getVocabLength(self):
        return len(self.vocabulary)

    def serialize(self):
        with open(self.config.readValue("vocabulary"), "wb") as f:
            pickle.dump(self, f)


class ResultsProcessor:

    def __init__(self):
        self.config = Config()

    def getDomainNames(self):
        return os.listdir("data/")

    def loadResults(self):
        path = self.config.readValue("results_path")
        with (open(path, "r")) as f:
            self.results = json.load(f)

    def stemInput(self):
        pass

    def lematizeInput(self):
        pass
    
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

    def readSelectedDomain(self, domain):
        path = self.config.readValue(PATH)
        self.domains = os.listdir(path)
        return pd.read_csv('aggregated/'+domain+'.csv', delimiter=',')

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
        print("Calculating jaro_winkler distance for words that occure once")
        
        self.serializeAnyDict(parsed_dict, "parsed_dict")
        self.parsed_dict = parsed_dict

    def getOccurances(self):
        new_occurances = {}
        for key,value in self.parsed_dict.items():
            new_occurances[key] = len(value)

        sorted_dict = [(i, new_occurances[i]) for i in sorted(new_occurances.keys())]
        self.newOccurances = sorted_dict

    def dumpOccurancesToJson(self):
        new_occurances = {}
        for key,value in self.parsed_dict.items():
            new_occurances[key] = len(value)

        sorted_dict = [(i, new_occurances[i]) for i in sorted(new_occurances.keys())]

        with(open("occurances.json", "w")) as f:
            json.dump(sorted_dict, f)
        self.newOccurances = sorted_dict

    def dumpParsedDictToJson(self):
        with(open("parsed.json", "w")) as f:
            json.dump(self.parsed_dict, f)
        

    def serializeDict(self):
        with(open("dict.pickl", "wb")) as f:
            pickle.dump(self.parsed_dict, f)

    def serializeAnyDict(self, obj, name):
        with(open("{}.pickl".format(name), "wb")) as f:
            pickle.dump(obj, f)

    def loadAnyDict(self, name):
        with(open("{}.pickl".format(name), "rb")) as f:
            return pickle.load(f)

    def loadDict(self):
        with(open("dict.pickl", "rb")) as f:
            self.parsed_dict = pickle.load(f)


    def autolabel(self, rects, ax):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

    def createPlot(self):
        r = 15
        labels = range(1,r,1)
        data = []

        for idx, label in enumerate(labels):
            for item in self.parsed_dict[label]:
                data.append(idx+1)
        counter=collections.Counter(data)
        n, bins, patches = plt.hist(data, align='left', rwidth=0.5, bins = range(1, r+1), facecolor='blue')
        plt.grid(True)
        plt.xlim(0, 15)
        plt.ylim(0, 250000)
        plt.xlabel('word occuracy')
        plt.ylabel('words count')
        path = self.config.readValue("plot_path")
        plt.savefig(path+"word_freqs.png")
        plt.show()

    def plotMeanLengthOfTextHistForSelectedDomain(self, domain):
        tokenizer = RegexpTokenizer(r'\w+')
        domain_data = self.readSelectedDomain(domain)
        data = [len(tokenizer.tokenize(x)) for idx, x in enumerate(domain_data["text"]) if isinstance(x,str)]
        n, bins, patches = plt.hist(data, bins = 50,rwidth=0.95 , facecolor='blue')
        plt.xlim(0, 600)
        plt.grid(True)
        plt.title("text length for domain: " + domain)
        plt.xlabel('text count')
        plt.ylabel('length')
        plt.savefig(self.config.readValue("plot_path")+"plotMeanLengthOfTextHistForSelectedDomain.png")
        plt.show()

    def plotMeanLengthOfThextHistForDataSet(self):
        tokenizer = RegexpTokenizer(r'\w+')
        path = self.config.readValue("data_set_path")
        domain_data = pd.read_csv(path, delimiter=',')
        data = [len(tokenizer.tokenize(x)) for idx, x in enumerate(domain_data["text"]) if isinstance(x,str)]
        n, bins, patches = plt.hist(data, bins = 50,rwidth=0.95, facecolor='blue')
        plt.xlim(0, 600)
        plt.grid(True)
        plt.title("text length for dataset")
        plt.xlabel('text count')
        plt.ylabel('length')
        path = self.config.readValue("plot_path")
        plt.savefig(path+"plotMeanLengthOfThextHistForDataSet")
        plt.show()

    def normalizedMeanLengthHistForDataSet(self):
        tokenizer = RegexpTokenizer(r'\w+')
        path = self.config.readValue("data_set_path")
        domain_data = pd.read_csv(path, delimiter=',')
        data = [len(tokenizer.tokenize(x)) for idx, x in enumerate(domain_data["text"]) if isinstance(x,str)]
        num_bins = 50
        n, bins, patches = plt.hist(data, num_bins, facecolor='blue', alpha=0.75)
        plt.xlim(0, 600)
        plt.grid(True)
        plt.show()

    def testMeanTextLenghtDifferenceBetweenAllDomains(self):
        print("\n")
        print("Testing mean text lengths difference between domains")
        print("Reading data")
        self.readData()
        print("Null hypothesis -> means of two populations are equal to each other")
        print("Calculating means")
        means = {}
        for domain in self.domains:
            means[domain] = FUNCTION_DEFINITIONS["lenghts"](self.frames[domain])
        print("Performing tests")
        accepted = 0
        rejected = 0
        tested = []
        for domain in self.domains:
            for second_domain in self.domains:
                if(domain != second_domain and not domain+second_domain in tested and not second_domain+domain in tested):
                    print("-----------***-----------")
                    print("Testing between {}, and {}".format(domain, second_domain))
                    ttest,pval = stats.ttest_ind(means[domain],means[second_domain])
                    print("ttest", ttest)
                    print("p-value", pval)
                    if pval <0.05:
                        print("we reject null hypothesis")
                        rejected += 1
                    else:
                        print("we accept null hypothesis")
                        accepted += 1
                    tested.append(domain+second_domain)

        print("Total accepted: {}, total rejected: {}".format(accepted, rejected))

    def testMeanTextLengthDiffBetweenPolarities(self):
        print("\n")
        print("-----------***-----------")
        print("Testing mean length difference between polarities")
        print("Reading data")
        path = self.config.readValue("data_set_path")
        data_set = pd.read_csv(path, delimiter=',')
        print("Null hypothesis -> means lenghts of text for polarity positive and negative are equal to each other")
        print("Calculating lengths")
        X = FUNCTION_DEFINITIONS["polarity_positive_lenghts"](data_set)
        Y = FUNCTION_DEFINITIONS["polarity_negative_lenghts"](data_set)
        print("Means for positive: {}, and negative: {}".format(np.mean(np.asarray(X)), np.mean(np.asarray(Y))))
        print("Performing tests")
        print("-----------***-----------")
        print("Testing between {}, and {}".format("positive", "negative"))
        ttest,pval = stats.ttest_ind(X,Y)
        print("ttest", ttest)
        print("p-value", pval)
        if pval <0.05:
            print("we reject null hypothesis")
        else:
            print("we accept null hypothesis")

    def anovaTestMeanTextLenghtOfDomains(self):
        print("\n")
        print("-----------***-----------")
        print("Anova testing mean length difference of texts between domains")
        print("Reading data")
        self.readData()
        print("Null hypothesis -> means of all populations are equal to each other")
        print("Alternative hypothesis -> there is a difference in means of the populations")
        print("Calculating means")
        means = {}
        for domain in self.domains:
            means[domain] = FUNCTION_DEFINITIONS["lenghts"](self.frames[domain])
        print("Performing tests")
        f, pval = stats.f_oneway(*means.values())
        print(f,pval)
        if pval <0.05:
            print("we reject null hypothesis")
        else:
            print("we accept null hypothesis")

    
    def getStopWordsInDataSet(self):
        self.parsed_dict = self.loadAnyDict("parsed_dict")
        minVal = max(self.parsed_dict.keys(), key=lambda x: int(x))
        lowestVal = [minVal]
        for i in range(minVal-1,minVal-15,-1):
            if i in self.parsed_dict:
                lowestVal.append(i)
        pom = []
        for y in lowestVal:
            for x in self.parsed_dict[y]:
                pom.extend(x)
        self.stopWords = pom
        return pom
        # return [*x for x in self.parsed_dict[y] for y in lowestVal]

    def getUniqueWordsInDataSet(self):
        self.unique = self.parsed_dict[1]
        return self.parsed_dict[1]

    def buildVocabulary(self):
        path = self.config.readValue("data_set_path")
        data_set = pd.read_csv(path)
        vocabs = [vocab for seq in data_set["text"] if isinstance(seq, str) for vocab in TOKENIZER.tokenize(seq)]
        vocab_count = Counter(vocabs)
        vocab_count = vocab_count.most_common(len(vocab_count))
        vocab_to_int = {word : index+2 for index, (word, count) in enumerate(vocab_count)}
        vocab_to_int.update({'__PADDING__': 0}) 
        vocab_to_int.update({'__UNKNOWN__': 1})
        vocabulary = Vocabulary(vocab_count, vocab_to_int)
        vocabulary.serialize()


if __name__ == "__main__":
    """
        args : 
            -plot -> results in histogram of word occurances
            -freq -> dumps 2 json files : occurances.json and parsed.json. it also serializes dict strucures
            -res -> does stuff ... mainly creates stuff ... and other stuff as well ...
            -calculate -> calculate occurances and freqs and dumps dict to pickle
    """
    rp = ResultsProcessor()
    if("-res" in sys.argv):
        rp.loadResults()
        rp.getValueFromAllDomains(sys.argv[2])
    elif("-calculate" in sys.argv):
        rp.calculateWordFreqs()
        rp.calculateWordOccurances()
        rp.serializeDict()
    elif("-freq" in sys.argv):
        rp.loadDict()
        rp.dumpOccurancesToJson()
        rp.serializeAnyDict(rp.newOccurances, "occurances_dict")
        rp.dumpParsedDictToJson()
    elif("-plot" in sys.argv):
        rp.loadDict() 
        rp.createPlot()
        rp.plotMeanLengthOfTextHistForSelectedDomain("Books")
        rp.plotMeanLengthOfThextHistForDataSet()
        # pprint(rp.parsed_dict[4151993])

    # - Test wether there is any statistically relevant difference in mean length (and other variables) of words between domains.
    elif("-test" in sys.argv):
        rp.testMeanTextLenghtDifferenceBetweenAllDomains()
        rp.testMeanTextLengthDiffBetweenPolarities()
        rp.anovaTestMeanTextLenghtOfDomains()

    elif("-preprocess" in sys.argv):
        rp.loadDict()
        rp.occurances = rp.loadAnyDict("occurances_dict")
        # rp.getOccurances()
        rp.getStopWordsInDataSet()
        rp.getUniqueWordsInDataSet()

    elif("-vocab") in sys.argv:
        rp.buildVocabulary()

    # Jakies pomocnicze skrypty do generowania rzeczy do pracy
    else:
        # df = pd.DataFrame(columns = ["Domain", "Mean text length (positive)", "Mean text length (negative)"])
        # temp = {}
        # rp.loadResults()
        # for idx, (key, value) in enumerate(zip(rp.results.keys(), rp.results.values())):
        #     temp[idx] = {"Domain": key, "Mean text length (positive)": value['averageTextLengthWhenPolarityPositiveWords'], "Mean text length (negative)": value['averageTextLengthWhenPolarityNegativeWords']}

        # results = pd.DataFrame.from_dict(temp, "index")
        # # print(results)
        # results.to_csv("pom.csv", sep = ";", index = False)

        temp = {}
        rp.loadResults()
        for idx, (key, value) in enumerate(zip(rp.results.keys(), rp.results.values())):
            temp[idx] = {"Domain": key, "Mean text length (positive)": value['averageTextLengthWhenPolarityPositiveChars'], "Mean text length (negative)": value['averageTextLengthWhenPolarityNegativeChars']}

        results = pd.DataFrame.from_dict(temp, "index")
        # print(results)
        results.to_csv("pom2.csv", sep = ";", index = False)