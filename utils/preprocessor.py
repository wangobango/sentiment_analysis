from .data_exploration import DataExplorer
from .data_loader import Loader, PolarityParser
from .config import Config
from console_progressbar import ProgressBar
from .data_exploration import Phrase
from .process_results import ResultsProcessor
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from autocorrect import Speller
from .config import Config
from console_progressbar import ProgressBar
import os
import pandas as pd
import re
import nltk
import multiprocessing as mp
import logging
import sys

PATH = "data_path"
LOGGER = logging.getLogger('preprocessor')

class Preprocessor:
    def __init__(self, numberOfProcesses = mp.cpu_count()):
        self.explorer = DataExplorer()
        self.resultsProcessor = ResultsProcessor()
        self.englishStopWords = set(stopwords.words('english')) 
        self.text = ''
        self.config = Config()
        self.flags = {
            'spelling': False,
            'stopWords': False,
            'lemmatize':False,
            'stem':False,
        }
        self.speller = Speller()
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = nltk.stem.SnowballStemmer('english')
        self.numberOfProcesses = numberOfProcesses


    def processSingleDataSetValue(self, value, polarity, output):
        word_tokens = word_tokenize(value)
        if(self.flags['spelling'] == True):
            lenghts = [self.reduce_lengthening(word) for word in word_tokens]
            word_tokens = [self.speller.autocorrect_word(word) for word in lenghts]
        if(self.flags['stopWords'] == True):
            word_tokens = [w for w in word_tokens if not w in self.englishStopWords]
        if(self.flags['lemmatize'] == True):
            word_tokens = [self.lemmatizer.lemmatize(word) for word in word_tokens]
        if(self.flags['stem'] == True):
            word_tokens = [self.stemmer.stem(word) for word in word_tokens]
        
        output.put([" ".join(word_tokens), polarity])

    def processChunk(self, list, output):
        for value in list:
            self.processSingleDataSetValue(value[0], value[1], output)


    def buildWithFlags(self):
        output = mp.Queue()
        offset = int(len(self.data_set)/self.numberOfProcesses)

        LOGGER.debug("Distributeing work to processes")
        processes = [mp.Process(target=self.processChunk, args=(zip(self.data_set[x*offset:(x+1)*offset], self.polarities[x*offset:(x+1)*offset]), output)) for x in range(self.numberOfProcesses)]        

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        LOGGER.debug("Calculation finished")
        results = pd.DataFrame([output.get() for p in processes])
        if(self.set):
            results.to_csv(self.config.readValue('processed_data_set'))
        else:
            results.to_csv(self.config.readValue('processed_test_set'))

    def setCorrectSPelling(self):
        self.flags['spelling'] = True
        return self

    def setStopWordsFlag(self):
        self.flags['stopWords'] = True
        return self

    def setLemmatizeFlag(self):
        self.flags['lemmatize'] = True
        return self

    def setStemmingFlag(self):
        self.flags['stemm'] = True
        return self

    def reduce_lengthening(self, text):
        pattern = re.compile(r"(.)\1{2,}")
        return pattern.sub(r"\1\1", text)

    @staticmethod
    def aggregateData():
        loader = Loader()
        loader.set_parser(PolarityParser())
        config = Config()
        topics = {}
        path = config.readValue(PATH)
        domains = os.listdir(path)
        pb = ProgressBar(total=int(len(domains)-1), prefix='Data parsing in progress',
                         suffix='', decimals=3, length=50, fill='X', zfill='-')
        frames = {}
        data = []

        for idx, topic in enumerate(domains):
            topics[topic] = []
            for item in os.listdir(path+topic):
                realPath = path + topic + "/" + item
                print(realPath)
                loader.set_path(realPath)
                data = loader.repair_file().load()

                if (len(data) > 0):
                    for sentance in data:
                        phrase = Phrase(*sentance.toArray())
                        topics[topic].append(phrase.toDictionary())
                else:
                    raise Exception('data length is 0')

            frames[topic] = pd.DataFrame(topics[topic])
            frames[topic].to_csv('aggregated/'+topic+'.csv')
            pb.print_progress_bar(idx)
        return frames

    def setText(self, text):
        self.text = text
        return self

    def removeStopWordsDatasetBased(self):
        stopWords = self.resultsProcessor.getStopWordsInDataSet()
        word_tokens = word_tokenize(self.text)
        stopped = [word for word in word_tokens if word in stopWords]
        filtered = [w for w in word_tokens if not w in stopWords]
        self.text = " ".join(filtered)
        return self

    def removeStopWordsEnglishCorpusBased(self):
        word_tokens = word_tokenize(self.text)
        filtered = [w for w in word_tokens if not w in self.englishStopWords]
        self.text = " ".join(filtered)
        return self

    def lemmatize(self):
        lemmatizer = WordNetLemmatizer() 
        word_tokens = word_tokenize(self.text)
        filtered = [lemmatizer.lemmatize(x) for x in word_tokens]
        self.text = " ".join(filtered)
        return self

    def stem(self):
        ps = nltk.stem.SnowballStemmer('english')
        word_tokens = word_tokenize(self.text)
        filtered = [ps.stem(x) for x in word_tokens]
        self.text = " ".join(filtered)
        return self

    def correctSpelling(self):
        speller = Speller()
        word_tokens = word_tokenize(self.text)
        lenghthening = [self.reduce_lengthening(word) for word in word_tokens]
        spelled = [speller.autocorrect_word(word) for word in lenghthening]
        self.text = " ".join(spelled)
        return self

    def build(self):
        return self.text

    """
        Removes \" and , TODO
    """
    def removePunctuationMarks(self):
        return self

    def preprocessDataSet(self):
        data_set = pd.read_csv(self.config.readValue('data_set_path'))
        self.data_set = data_set['text']
        self.polarities = data_set['polarity']
        self.set = True
        return self

    def preprocessTestSet(self):
        data_set = pd.read_csv(self.config.readValue('test_set_path'))
        self.data_set = data_set['text']
        self.polarities = data_set['polarity']
        self.set = False
        return self

if __name__ == "__main__":
    if "--log" in sys.argv:
        logging.basicConfig(level=logging.INFO)
    """
        @prerequisites:
            directory 'processed' created in root directory of the project
        @params:
            text -> text to be processed, given as a single string
        @returns:
            text processed after applying selected build steps
        Usage:
            prep = Preprocessor()
            prep.setText(text).correctSpelling().{...}.build()

        Static:
            It's possible to only use static method to aggregate data like:
                from utils.preprocessor import Preprocessor
                Preprocessor.aggregateData()
    """
    prep = Preprocessor()
    # Example:
    # data = pd.read_csv('./data_set/data_set.csv', nrows=10)
    # example = data['text'][1]
    # print(example)
    # print("---------***----------")
    # text = prep.setText(example).removeStopWordsDatasetBased().build()
    # print(text)

    prep.preprocessDataSet().setStemmingFlag().setLemmatizeFlag().setStopWordsFlag().setCorrectSPelling().buildWithFlags()
    prep.preprocessTestSet().setStemmingFlag().setLemmatizeFlag().setStopWordsFlag().setCorrectSPelling().buildWithFlags()
    
