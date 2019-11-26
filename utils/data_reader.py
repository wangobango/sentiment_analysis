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


    def read_data_set(self, shuffle=True):
        LOGGER.info("Reading data set...")
        return self.read_set(self.data_set_file_path)

    def read_test_set(self, shuffle=True):
        LOGGER.info("Reading test set...")
        return self.read_set(self.test_set_file_path)   

    def read_set(self, data_set_file, shuffle=True):
        dset = pd.read_csv(data_set_file)
        if shuffle:
            dset = utils.shuffle(dset)
            dset.reset_index(inplace=True, drop=True)
        return dset