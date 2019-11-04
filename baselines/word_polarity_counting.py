from dictionaries.dictionary import Dictionary
from config import Config
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import pandas as pd

TEST_SET_PATH = "test_set_path"
class WordPolarityCounting:
    
    def __init__(self):
        self.config = Config()
        self.lemmatizer = WordNetLemmatizer()
        self.dictionary = Dictionary()
        self.read_data_set()
        self.process()

    def read_data_set(self):
        data_set_file = self.config.readValue(TEST_SET_PATH)
        self.data_set = pd.read_csv(data_set_file)

    def process(self):
        tokenizer = RegexpTokenizer(r'\w+')
        for i, row in self.data_set.iterrows():
            if i == 1:
                print(row['text'])
                print(sum(1 if self.dictionary.get_word_polarity_numerical_value(token, False) == "empty" else 0 for token in tokenizer.tokenize(row['text'])))
                polarity = sum([self.get_word_polarity_numerical_value(token) for token in tokenizer.tokenize(row['text'])])
                print(row['polarity'])
                print(polarity)
                # for token in tokenizer.tokenize(row['text']):
                #     print("Token: \"{}\"".format(token.lower()))
                #     print("polarity: {}".format(self.dictionary.get_word_polarity_numerical_value(token)))
                break
    def get_word_polarity_numerical_value(self, token):
        polarity = self.dictionary.get_word_polarity_numerical_value(self.lemmatizer.lemmatize(token, pos="v"))
        # polarity = self.dictionary.get_word_polarity_numerical_value(token, False)
        return float(polarity if not polarity == "empty" else 0)
    
if __name__ == "__main__":
    wpc = WordPolarityCounting()
