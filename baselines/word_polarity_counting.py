from dictionaries.senticnet_dictionary import Dictionary
from utils.data_evaluator import Evaluator
from utils.config import Config
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn import utils
from pickle import dump, load
import logging
import time
import sys
import multiprocessing as mp
import numpy as np
import pandas as pd

DATA_SET_PATH = "data_set_path"
TEST_SET_PATH = "test_set_path"
MODEL_PROP = "word_polarity_counting_model"
LOGGER = logging.getLogger("WordPolarityCounting")

class WordPolarityCounting:
    
    def __init__(self):
        self.config = Config()
        self.evaluator = Evaluator()
        self.lemmatizer = WordNetLemmatizer()
        self.dictionary = Dictionary()
        self.read_data_sets()
        self.border_value = None

    def read_data_sets(self):
        data_set_file = self.config.readValue(DATA_SET_PATH)
        test_set_file = self.config.readValue(TEST_SET_PATH)
        LOGGER.info("Reading data set...")
        self.data_set = self.read_set(data_set_file)
        LOGGER.info("Reading test set...")
        self.test_set = self.read_set(test_set_file)

    def read_set(self, data_set_file):
        dset = utils.shuffle(pd.read_csv(data_set_file))
        dset.reset_index(inplace=True, drop=True)
        return dset

    """
    Function "teaches" model based on preloded data set - it calculates border value which later determines text polarity.
    If "parallel" is set to True, function uses multiprocessing in calculations.
    """
    def teach(self, parallel=False):
        start_time = time.time()
        df = self.data_set.tail(20)[['polarity', 'text']].copy()
        LOGGER.info("Calculating polarity num values...")
        if parallel:
            p = mp.Pool(mp.cpu_count())
            df['polarity_num_val'] = p.map(self.get_text_polarity_num_val, df['text'])
        else:
            df['polarity_num_val'] = df.apply(lambda row: self.get_text_polarity_num_val(row['text']), axis=1)
        self.border_value = df['polarity_num_val'].mean()
        LOGGER.info("Calulation took: {} s".format(time.time() - start_time))
        LOGGER.info("Dumpling model into file...")
        with open(self.config.readValue(MODEL_PROP), 'wb') as model:
            dump(self.border_value, model)

    """
    Predicts text polarity from preloaded test set based on previously calculated border value
    """
    def evaluate(self):
        if self.border_value == None:
            LOGGER.info("Reading model from file...")
            with open(self.config.readValue(MODEL_PROP), 'rb') as model:
                self.border_value = load(model)
        LOGGER.info('Evaluating model...')
        start_time = time.time()
        df_test = self.test_set.tail(10)[['polarity', 'text']].copy()
        df_test = self.test_set[['polarity', 'text']].copy()
        df_test['predicted'] = df_test.apply(lambda row: self.predict_polarity(row['text']), axis=1)
        LOGGER.info("Evaluation took: {} s".format(time.time() - start_time))
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
    
    """
    Predicts single text polarity based on previoulsy calculated border value
    """
    def predict_polarity(self, text):
        return 'positive' if self.get_text_polarity_num_val(text) > self.border_value else 'negative'

    """
    Returns numerical value of token polarity. 
    Mappings:
    -1 if token polarity is 'negative'
     0 if token polarity was not found in dictionary
     1 if token polarity is 'positive'
    """
    def get_word_polarity_numerical_value(self, token):
        polarity = self.dictionary.get_word_polarity(self.lemmatizer.lemmatize(token, pos="v"), False)
        return float(0 if polarity == "empty" else (-1 if polarity == "negative" else 1))
        
    
if __name__ == "__main__":
    if "--help" in sys.argv:
        print("Script for training and evaluation baseline model. Script can be used with following comands:\n" + \
            " teach - trains the model based on preloaded data\n evaluate - evaluates model based on previosuly trained model" + \
            "\nTo get more information of script process add --log to command\nTo train model parallelly use --parallel" + \
            "\nexample usage\n'python3 -m baselines.word_polarity_counting teach --log --parallel'")
        exit(0)
    if "--log" in sys.argv:
        logging.basicConfig(level=logging.INFO)
    parallel = "--parallel" in sys.argv
    if "teach" in sys.argv or "evaluate" in sys.argv:
        wpc = WordPolarityCounting()
    else:
        print("\nNothing to do...\nPlese use this script with either 'tech' command or 'evaluate'\n For more info type:\n\
            'Python3 -m baselines.word_polarity_counting --help'")
    if "teach" in sys.argv:
        wpc.teach(parallel)
    if "evaluate" in sys.argv:
        wpc.evaluate()
