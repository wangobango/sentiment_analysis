import pandas as pd
from utils.config import Config

class LstmModel:
    def __init__(self):
        self.config = Config()

    def getSplittedData(self):
        data_set = pd.read_csv(self.config.readValue('processed_data_set'))
        test_set = pd.read_csv(self.config.readValue('processed_test_set'))
        
        X_train, Y_train = data_set['text'], data_set['polarity']
        X_test, Y_test = test_set['text'], test_set['polarity']

        return (X_train, Y_train), (X_test, Y_test)

    