from utils.data_reader import DataReader
from utils.config import Config
import pandas as pd
LABEL_MAP = {'negative': 0, 'positive': 1}

# Class for prepering csv data into tsv data
class BertDatapreparator():
    def __init__(self):
        self.dataReader = DataReader()
        self.config = Config()
        self.labelMap = LABEL_MAP

    def prepareDevset(self, amount=None):
        dataSet = self.dataReader.read_data_set()
        if amount is not None:
            dataSet = dataSet[:amount]
        dev_df_bert = pd.DataFrame({
            'id': range(len(dataSet)),
            'label':dataSet['polarity'].replace(self.labelMap),
            'alpha':['a']*dataSet.shape[0],
            'text': dataSet['text'].replace(r'\n', ' ', regex=True)
        })
        dev_df_bert.to_csv(self.config.readValue('bert_dev_set'), sep='\t', index=False, header=False)

    def prepareTestset(self, amount=None):
        testSet = self.dataReader.read_data_set()
        if amount is not None:
            testSet = testSet[:amount]
        test_df_bert = pd.DataFrame({
            'id': range(len(testSet)),
            'label':testSet['polarity'].replace(self.labelMap),
            'alpha':['a']*testSet.shape[0],
            'text': testSet['text'].replace(r'\n', ' ', regex=True)
        })
        test_df_bert.to_csv(self.config.readValue('bert_test_set'), sep='\t', index=False, header=False)
        print(test_df_bert.head())

if __name__ == "__main__":
    preparator = BertDatapreparator()
    preparator.prepareDevset(100)
    preparator.prepareTestset(100)
