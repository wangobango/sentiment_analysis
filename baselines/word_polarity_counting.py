from dictionaries.dictionary import Dictionary
from data_evaluator import Evaluator
from config import Config
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn import utils
from itertools import repeat
from sklearn.linear_model import LogisticRegression
import time
import multiprocessing as mp
import numpy as np
import pandas as pd

DATA_SET_PATH = "data_set_path"
TEST_SET_PATH = "test_set_path"

class WordPolarityCounting:
    
    def __init__(self):
        self.config = Config()
        self.evaluator = Evaluator()
        self.lemmatizer = WordNetLemmatizer()
        self.dictionary = Dictionary()
        self.read_data_sets()

    def read_data_sets(self):
        data_set_file = self.config.readValue(DATA_SET_PATH)
        test_set_file = self.config.readValue(TEST_SET_PATH)
        self.data_set = self.read_set(data_set_file)
        self.test_set = self.read_set(test_set_file)

    def read_set(self, data_set_file):
        dset = utils.shuffle(pd.read_csv(data_set_file))
        dset.reset_index(inplace=True, drop=True)
        return dset

    """
    Function processes
    """
    def process(self, mulitprocess=False, log=False):
        start_time = time.time()
        p = mp.Pool(mp.cpu_count())
        df = self.data_set[['polarity', 'text']].copy()
        if log: print("Calculating polarity num values")
        if mulitprocess:
            df['polarity_num_val'] = p.map(self.get_text_polarity_num_val, df['text'])
        else:
            df['polarity_num_val'] = df.apply(lambda row: self.get_text_polarity_num_val(row['text']), axis=1)

        calculating_time = time.time()
        if log:
            print("Calculating took: {} s".format(calculating_time - start_time))
            print("Predicting result...")
        breakpoint = df['polarity_num_val'].mean()
        df_test = df = self.test_set[['polarity', 'text']].copy()
        df_test['predicted'] = df_test.apply(lambda row: self.predict_polarity(row['text'], breakpoint), axis=1)
        if log: print("Predicting took: {} s".format(time.time() - calculating_time))
        self.evaluator.evaluate(df_test['polarity'], df_test['predicted'])

    """
    Function which calucates polatity value / score for whole text
    """
    def get_text_polarity_num_val(self, text):
        tokenizer = RegexpTokenizer(r'\w+')
        if isinstance(text, str):
            tokens = tokenizer.tokenize(text)
            positives, negatives = 0, 0
            for token in tokens:
                polarity = self.dictionary.get_word_polarity(self.lemmatizer.lemmatize(token, pos="v"), False)
                if polarity == 'positive': positives += 1
                elif polarity == 'negative': negatives += 1
            
            return positives / max(negatives, 1)
        else:
            return 0

    # def processLogisitcRegression(self, mulitprocess=False):
    #     logreg = LogisticRegression()
    #     start_time = time.time()
    #     p = mp.Pool(mp.cpu_count())
    #     df = self.data_set.head(5)[['polarity', 'text']].copy()
    #     if mulitprocess:
    #         df['polarity_num_val'] = p.map(self.get_text_polarity_num_val, df['text'])
    #         df['polarity'] = p.map(self.maps_polarity_tu_num, df['polarity'])
    #     else:
    #         df['polarity_num_val'] = df.apply(lambda row: self.get_text_polarity_num_val(row['text']), axis=1)
    #         df['polarity'] = df.apply(lambda x:  self.maps_polarity_tu_num(x['polarity']), axis=1)
    #     calculating_time = time.time()
    #     print("Calculating took: {} s".format(calculating_time - start_time))
    #     print("Fitting model")
    #     logreg.fit(df['polarity_num_val'], df['polarity'])
    
    def predict_polarity(self, text, brakepoint):
        return 'positive' if self.get_text_polarity_num_val(text) > brakepoint else 'negative'

    def maps_polarity_tu_num(self, polarity):
        return 1 if polarity == 'positive' else 0

    def get_word_polarity_numerical_value(self, token):
        polarity = self.dictionary.get_word_polarity(self.lemmatizer.lemmatize(token, pos="v"), False)
        return float(0 if polarity == "empty" else (-1 if polarity == "negative" else 1))

    def get_word_polarity(self, token):
        return self.dictionary.get_word_polarity(token, False)
        
    
if __name__ == "__main__":
    wpc = WordPolarityCounting()
