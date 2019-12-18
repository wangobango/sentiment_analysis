import pandas as pd
import numpy as np
import pickle
import logging
import sys
import spacy
import copy

# from utils.config import Config
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, TimeDistributed, Activation
from nltk.tokenize import word_tokenize 

EMBEDDING_LENGTH = 300
LOGGER = logging.getLogger('lstm-model')

class LstmModel:
    def __init__(self, batch_size, sequence_length, epochs, hidden_size, output_size, input_length, vocabulary):
        # self.config = Config()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_length = input_length
        self.vocabulary = vocabulary
        self.nlp = spacy.load('en_core_web_md')


    def getSplittedData(self):
        # data_set = pd.read_csv(self.config.readValue('processed_data_set'))
        # test_set = pd.read_csv(self.config.readValue('processed_test_set'))
        data_set = pd.read_csv('processed/data_set.csv')
        test_set = pd.read_csv('processed/test_set.csv')

        X_train, Y_train = data_set['text'], data_set['polarity']
        X_test, Y_test = test_set['text'], test_set['polarity']

        LOGGER.debug("Data acquired")
        return (X_train, Y_train), (X_test, Y_test)

    def addPadding(self, sequence):
        if(len(sequence) < self.sequence_length):
            for _ in range(self.sequence_length - len(sequence)):
                sequence.append(np.zeros((self.sequence_length,)))
        return sequence

    def initModel(self):
        LOGGER.debug("Initializing model")
        self.model = Sequential()
        # self.model.add(Embedding(self.vocabulary, self.hidden_size, input_length=self.input_length))
        self.model.add(Embedding)
        self.model.add(LSTM(
                input_shape=(self.sequence_length, self.hidden_size),
                stateful=False,
                units=self.hidden_size,
                return_sequences=True))
        self.model.add(LSTM(self.hidden_size, return_sequences=True, stateful=False))
        self.model.add(LSTM(self.hidden_size, return_sequences=True, stateful=False))                
        self.model.add(TimeDistributed(Dense(self.hidden_size)))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='mse', optimizer='adam')
        
        LOGGER.debug("Initialized model")

    def train(self):
        LOGGER.debug("Beginning to train")
        X_train, Y_train,X_test, Y_test = self.tempEmbedding()
        print(len(X_train))
        self.model.fit(X_train, Y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_data=(X_test, Y_test), shuffle=False)

    def serializeModel(self):
        with open(self.config.readValue("lstm_model_path"), "wb") as f:
            pickle.dump(self.model, f)
    
    def loadModel(self):
        with open(self.config.readValue("lstm_model_path"), "rb") as f:
            self.model = pickle.load(f)

    def summary(self):
        print(self.model.summary())

    def tempEmbedding(self):
        (X_train, Y_train), (X_test, Y_test) = self.getSplittedData()
        resultX_train, resultY_train, resultX_test, resultY_test = ([],[],[],[])
        sequenceTrain = []
        sequenceTest = []
        for items in zip(X_train, Y_train, X_test, Y_test):
            for value in zip(items[0], items[1]):
                for word in word_tokenize(value[0]):
                    sequenceTrain.append([self.nlp(word).vector, 1 if value[1] == 'positive' else 0])

            for value in zip(items[2], items[3]):
                for word in word_tokenize(value[0]):
                    sequenceTest.append([self.nlp(word).vector, 1 if value[1] == 'positive' else 0])


            resultX_train.append(copy.copy(sequenceTrain))
            resultX_test.append(copy.copy(sequenceTest))
            # for value in word_tokenize(items[0]):
            #     resultX_train.append(self.nlp(value).vector)
            #     if(items[1] == 'positive'):
            #         resultY_train.append(1)
            #     else:
            #         resultY_train.append(0)
            # for value in word_tokenize(items[2]):
            #     resultX_test.append(self.nlp(value).vector)
            #     if(items[3] == 'positive'):
            #         resultY_test.append(1)
            #     else:
            #         resultY_test.append(0)
        # resultX_train = np.asarray(resultX_train)
        # resultX_train = np.reshape(resultX_train, (self.sequence_length, EMBEDDING_LENGTH))
        
        # resultX_test = np.asarray(resultY_test)
        # resultX_test = np.reshape(resultX_test, (self.sequence_length, EMBEDDING_LENGTH))

        return resultX_train, resultY_train, resultX_test, resultY_test


if __name__ == "__main__":
    if("--log" in sys.argv):
        logging.basicConfig(level=logging.DEBUG)
    lstm = LstmModel(20, 30, 3, 100, 3, EMBEDDING_LENGTH, 30)
    lstm.initModel()
    lstm.summary()
    data = lstm.tempEmbedding()
    print('dupa')
    # lstm.train()