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
from dask.distributed import Client, progress
import os
import pandas as pd
import re
import nltk
import dask.multiprocessing
import dask
import dask.bag as db

PATH = "data_path"


class Preprocessor:
    def __init__(self):
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

    def processSingleDataSetValue(self, value):
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
        
        return " ".join(word_tokens)


    def buildWithFlags(self):
        client = Client()
        print("Creating Dask bag.")
        bag = db.from_sequence(self.data_set[1:1000])
        print("Applying function to data set")
        proc = bag.map(lambda x: self.processSingleDataSetValue(x))
        client.compute(proc, num_workers=8, scheduler='processes')
        print("Computing finished")
        return proc

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
        return frames
            # print("Done topic: {}, {} / {}".format(topic, idx, len(domains)))

    def setText(self, text):
        self.text = text
        return self

    # TODO zwraca GÓWNO jak narazie :)
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
        Removes \" and ,
    """
    def removePunctuationMarks(self):
        return self



    # TODO , use list as an input and use jobs library to concat it !!!! + make build steps as flags instead of actuall processing
    def preprocessDataSet(self):
        self.data_set = pd.read_csv(self.config.readValue('data_set_path'))['text']
        return self

    def preprocessTestSet(self):
        pass

if __name__ == "__main__":
    """
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
    # data = pd.read_csv('./data_set/data_set.csv', nrows=10)
    # example = data['text'][1]
    # print(example)
    # print("---------***----------")
    # text = prep.setText(example).removeStopWordsDatasetBased().build()
    # print(text)

    # TODO usunięcie znakow nowej lini
    # BUG usuwanie znaków interpunkcyjnych za każdym jebanym RAZEM 
    dupa = prep.preprocessDataSet().setStemmingFlag().setLemmatizeFlag().setStopWordsFlag().setCorrectSPelling().buildWithFlags()
    print(dupa)
    print(type(dupa))
