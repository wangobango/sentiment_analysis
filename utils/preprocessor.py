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
from .word2vec_mapper import Word2VecMapper
import time
import os
import pandas as pd
import re
import nltk
import multiprocessing as mp
import logging
import sys
import copy
import numpy as np
import spacy

PATH = "data_path"
LOGGER = logging.getLogger('preprocessor')

class Preprocessor:
    def __init__(self, numberOfProcesses = mp.cpu_count()-1, optional_length = None):
        if "--log" in sys.argv:
            logging.basicConfig(level=logging.DEBUG)
        if "-embedding" in sys.argv:
            self.EMBEDDING = True
        else:
            self.EMBEDDING = False
        self.EMBEDDING_LENGTH = 300
        self.SEQUENCE_LENGTH = 100
        self.TIMEOUT = 15
        self.OVERHEAD_TIMEOUT = 45
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
        self.mapper = Word2VecMapper()
        self.optional_length = optional_length
        self.config = Config()


    def processSingleDataSetValue(self, value, polarity, id,  output, objs, flags):
        word_tokens = word_tokenize(value)
        if(flags['spelling'] == True):
            lenghts = [self.reduce_lengthening(word) for word in word_tokens]
            word_tokens = [objs['speller'].autocorrect_word(word) for word in lenghts]
        if(flags['stopWords'] == True):
            word_tokens = [w for w in word_tokens if not w in objs['stop']]
        if(flags['lemmatize'] == True):
            word_tokens = [objs['lemmatizer'].lemmatize(word) for word in word_tokens]
        if(flags['stem'] == True):
            word_tokens = [objs['stemmer'].stem(word) for word in word_tokens]


        if(self.EMBEDDING):
            counter = 0
            for word in word_tokens:
                output.put([id, objs['mapper'].word2vec(word), 1 if polarity == 'positive' else 0])
                counter += 1
            # if(counter < self.SEQUENCE_LENGTH):
            #     for _ in range(0, self.SEQUENCE_LENGTH - counter):
            #         output.put([id, np.zeros((self.EMBEDDING_LENGTH,)), 1 if polarity == 'positive' else 0])
        else:
            output.put([id, " ".join(word_tokens), 1 if polarity == 'positive' else 0])

    def processChunk(self, list, output, procId):
        objs = {
            'speller' : Speller(),
            'lemmatizer' : WordNetLemmatizer(),
            'stemmer' : nltk.stem.SnowballStemmer('english'),
            'mapper' : Word2VecMapper(),
            'stop' : set(stopwords.words('english'))
        }
        flags = copy.copy(self.flags)
        for idx,value in enumerate(list):
            if(idx % 10 == 0):
                LOGGER.debug('{}, done {}/{}'.format(procId, idx+1, int(len(self.data_set)/self.numberOfProcesses)))
            self.processSingleDataSetValue(value[0], value[1], value[2], output, objs, flags)
        LOGGER.debug("{}, finished processing".format(procId))
        # output.cancel_join_thread()

    def buildWithFlags(self):
        output = mp.Queue()
        offset = int(len(self.data_set)/self.numberOfProcesses)
        results = pd.DataFrame(columns = ['id', 'embedding', 'polarity'])

        LOGGER.debug("Distributeing work to processes")
        processes = [mp.Process(target=self.processChunk, args=(zip(self.data_set[x*offset:(x+1)*offset], self.polarities[x*offset:(x+1)*offset], self.ids[x*offset:(x+1)*offset]), output, x)) for x in range(self.numberOfProcesses)]        
        self.processes = processes

        for idx,p in enumerate(processes):
            LOGGER.debug("Staring process {}/{}".format(idx+1, self.numberOfProcesses))
            p.start()


        if(self.EMBEDDING):
            numberOfItems = offset * self.numberOfProcesses * self.SEQUENCE_LENGTH
        else:
            numberOfItems = offset * self.numberOfProcesses
        elapsed = 0
        counter = 0
        time.sleep(self.OVERHEAD_TIMEOUT)
        start_time = time.time()
        LOGGER.debug("Consumeing output")
        # while counter < numberOfItems:
        while True:
            if(not output.empty()):
                if(counter % 10 == 0):
                    LOGGER.debug("Output size: {}".format(output.qsize()))
                elapsed = 0
                start_time = time.time()
                counter += 1
                value = output.get()
                results = results.append({'id': value[0], 'embedding' : value[1], 'polarity': value[2] }, ignore_index = True)
            else:
                current_time = time.time()
                elapsed += current_time - start_time
                start_time = current_time
            
            if(elapsed >= self.TIMEOUT):
                LOGGER.debug("Timeout! Output size is {}, time elapsed: {}".format(output.qsize(), elapsed))
                break


        LOGGER.debug("Joining threads")

        for idx,p in enumerate(processes):
            LOGGER.debug("Joining process {}/{}".format(idx+1, self.numberOfProcesses))
            p.join()

        LOGGER.debug("Calculation finished")
        # results = pd.DataFrame([output.get() for p in processes])
        # results = pd.DataFrame(columns = ['embedding', 'polarity'])
        while not output.empty():
            value = output.get()
            results = results.append({ 'embedding' : value[0], 'polarity': value[1] }, ignore_index = True)
        if(self.set):
            results.to_csv(self.config.readValue('processed_data_set'), sep = ";")
        else:
            results.to_csv(self.config.readValue('processed_test_set'), sep = ";")

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
        if(self.optional_length != None):
            data_set = data_set[0:self.optional_length]
        self.data_set = data_set['text']
        self.polarities = data_set['polarity']
        self.ids = data_set['id']
        self.set = True
        return self

    def preprocessTestSet(self):
        data_set = pd.read_csv(self.config.readValue('test_set_path'))
        if(self.optional_length != None):
            data_set = data_set[0:self.optional_length]
        self.data_set = data_set['text']
        self.polarities = data_set['polarity']
        self.ids = data_set['id']
        self.set = False
        return self

if __name__ == "__main__":
    # TODO returns batched size with self.self.EMBEDDINGs, need to add padding
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
    prep = Preprocessor(numberOfProcesses=6, optional_length=48*6*6)
    # Example:
    # data = pd.read_csv('./data_set/data_set.csv', nrows=10)
    # example = data['text'][1]
    # print(example)
    # print("---------***----------")
    # text = prep.setText(example).removeStopWordsDatasetBased().build()
    # print(text)


    # TODO find other spelling correcter
    # TODO threads not joining afther being finished
    # TODO for some reason there are always the same amount of rows saved lol . fix it 
    prep.preprocessDataSet().correctSpelling().setLemmatizeFlag().setStopWordsFlag().buildWithFlags()
    prep.preprocessTestSet().correctSpelling().setLemmatizeFlag().setStopWordsFlag().buildWithFlags()
    
