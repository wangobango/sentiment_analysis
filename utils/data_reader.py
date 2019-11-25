import logging
import pandas as pd
from utils.config import Config
from sklearn import utils

LOGGER = logging.getLogger("DataReader")
DATA_SET_PATH = "data_set_path"
TEST_SET_PATH = "test_set_path"

class DataReader:

    def __init__(self): 
        self.config = Config()
        self.data_set_file_path = self.config.readValue(DATA_SET_PATH)
        self.test_set_file_path = self.config.readValue(TEST_SET_PATH)


    def read_data_set(self):
        LOGGER.info("Reading data set...")
        self.data_set = self.read_set(self.data_set_file_path)

    def read_test_set(self):
        LOGGER.info("Reading test set...")
        self.test_set = self.read_set(self.test_set_file_path)   

    def read_set(self, data_set_file):
        dset = utils.shuffle(pd.read_csv(data_set_file))
        dset.reset_index(inplace=True, drop=True)
        return dset