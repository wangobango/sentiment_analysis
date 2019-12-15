from .data_loader import Loader
from .data_loader import PolarityParser
from .config import Config
import pandas as pd
import spacy



class Word2VecMapper:

    def __init__(self):
        self.nlp = spacy.load('en_core_web_md')
        # self.target_train_set = open("/home/jacek/Downloads/inzynierka/projekt/sentiment_analysis/data_set/embedded.csv", "a")
        self.target_train_set = None
        # self.target_test_set = open("/home/jacek/Downloads/inzynierka/projekt/test_set/embedded.csv", "w")

    def set_target(self, path):
        self.target_train_set = open(path, "a")

    def read(self, lines = 0, path):

        f = open(path, "r")
        try:
            i =0
            while True:
                i+=1
                print(str(i/700000) + "%")
                row = f.readline()
                if(row == ''):
                    break
                self.put_row(row)
        finally:
            f.close()

    def put_row(self, row):
        if len(row) < 10:
            return
        srow = row.split("\t")
        # if(len(srow) > 5):
        #     print(row)
        #     print("\n")

        wrow = []
        wrow.append(srow[0])
        wrow.append(srow[1])
        if srow[2] == 'negative':
            wrow.append("0")
        else:
            wrow.append("1")
        
        self.put_particular_words(wrow, srow[4])

    def put_particular_words(self, text_info, text):
        words = text.split(" ")
        # dic = self.nlp(text)
        for w in words[:-1]:  #TODO iterate full list after pre-processing process
            warray = []
            # embedding = self.find_embeding_.as_str(w.replace('"', ''))
            embedding = self.nlp(w).vector
            embedding = embedding.tolist()
            embedding = ';'.join(map(str, embedding))
            # if(embedding == 0):
            #     continue
            warray.append(text_info[0])
            warray.append(text_info[1])
            warray.append(text_info[2])
            warray.append(w.replace('"', ''))
            warray.append(embedding)
            # write = 
            self.target_train_set.write(";".join(warray) + "\n")

    def word2vec(self, word):
        return self.nlp(word).vector
    
    def find_embeding_as_str(self, word):
        f = open('/home/jacek/Downloads/inzynierka/projekt/sentiment_analysis/sentic2vec.csv', "r")
        while True:
            try:
                row = f.readline()
            except:
                continue
            if(row == ''):
                break
            
            row = row.replace('"', '')
            row = row.split(",")
            row[-1] = row[-1].replace("\n", '')
            w = row[0]
            if(str(w) == str(word)):
                f.close()
                print("we found word: " + word)
                return ";".join(row[1:])

        print("Error: word '" + word + "' not found in embedings")
        f.close()
        return 0

# if __name__ == "__main__":  
#     w2v = Word2VecMapper()
#     w2v.read()
    # w2v.find_embeding_as_str("people")