from utils.data_reader import DataReader
from utils.config import Config
import pandas as pd

# Class for prepering csv data into tsv data
class BertDatapreparator():
    def __init__(self):
        self.dataReader = DataReader()
        self.config = Config()
        self.labelDict = {'negative': 0, 'positive': 1}

    def prepareDevset(self):
        dataSet = self.dataReader.read_data_set()
        dev_df_bert = pd.DataFrame({
            'id': range(len(dataSet)),
            'label':dataSet['polarity'].replace(self.labelDict),
            'alpha':['a']*dataSet.shape[0],
            'text': dataSet['text'].replace(r'\n', ' ', regex=True)
        })
        dev_df_bert.to_csv(self.config.readValue('bert_dev_set'), sep='\t', index=False, header=False)
        print(dev_df_bert.head())

    def prepareTestset(self):
        testSet = self.dataReader.read_data_set()
        test_df_bert = pd.DataFrame({
            'id': range(len(testSet)),
            'label':testSet['polarity'].replace(self.labelDict),
            'alpha':['a']*testSet.shape[0],
            'text': testSet['text'].replace(r'\n', ' ', regex=True)
        })
        test_df_bert.to_csv(self.config.readValue('bert_test_set'), sep='\t', index=False, header=False)
        print(test_df_bert.head())

if __name__ == "__main__":
    preparator = BertDatapreparator()
    preparator.prepareDevset()
    preparator.prepareTestset()
