CUDA_LAUNCH_BLOCKING="1"
import torch.utils.data.sampler as splr
import numpy as np
import pandas as pd
import logging
import sys
import pickle
import torch.nn as nn
import torch
import torch.nn.functional as functional
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.config import Config
from utils.process_results import Vocabulary
from nltk.tokenize import RegexpTokenizer
from console_progressbar import ProgressBar
from utils.data_evaluator import Evaluator
from utils.preprocessor import Preprocessor

TOKENIZER = RegexpTokenizer(r'\w+')
LOGGER = logging.getLogger('lstm_model')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(threshold=sys.maxsize)

"""
    Params start
"""
set_count = 100000
epochs = 1
counter = 0
learning_rate = 0.0001
weight_decay = 0.005
momentum = 0.9
clip = 5
embedding_dim = 150
hidden_dim = 300
output_size = 1
n_layers = 2
batch_size = 50

class DataSampler(object):
    
    def __init__(self, input_tensor, input_lengths, labels_tensor, batch_size, sequence_lenght = 2665):
        self.input_tensor = input_tensor
        self.input_lengths = input_lengths
        self.labels_tensor = labels_tensor
        self.batch_size = batch_size
        self.sequence_length = 2665

        self.sampler = splr.BatchSampler(splr.RandomSampler(self.labels_tensor), self.batch_size, False)
        self.sampler_iter = iter(self.sampler)

    def __iter__(self):
        self.sampler_iter = iter(self.sampler) 
        return self

    def _next_index(self):
        return next(self.sampler_iter)

    def __len__(self):
        return len(self.sampler)

    def __next__(self):
        index = self._next_index()

        subset_input_tensor = self.input_tensor[index]
        subset_input_lengths = self.input_lengths[index]
        subset_labels_tensor = self.labels_tensor[index]

        subset_input_lengths, perm_idx = subset_input_lengths.sort(0, descending=True)
        subset_input_tensor = subset_input_tensor[perm_idx]
        subset_labels_tensor = subset_labels_tensor[perm_idx]

        return subset_input_tensor, subset_input_lengths, subset_labels_tensor


    @staticmethod
    def aggregate():
        conf = Config()
        data = pd.read_csv(conf.readValue('processed_data_set'), sep = ";")
        test = pd.read_csv(conf.readValue('processed_test_set'), sep = ";")

        data = data.groupby(['id','polarity'])['embedding'].apply(list).reset_index(name='sequence')
        test = test.groupby(['id','polarity'])['embedding'].apply(list).reset_index(name='sequence')
        
        return data, test 


class PolarityGRU(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_dim, output_size, n_layers, drop_lstm=0.1, drop_out = 0.1):
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_lstm, batch_first=True)
        self.dropout = nn.Dropout(drop_out)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, seq_lengths):  
        embedded_seq_tensor = self.embedding(x)
        packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)

        packed_output, (ht, ct) = self.gru(packed_input, None)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
       
        last_idxs = (input_sizes - 1).to(device) 
        output = torch.gather(output, 1, last_idxs.view(-1, 1).unsqueeze(2).repeat(1, 1, self.hidden_dim)).squeeze() 

        output = self.dropout(output)
        output = self.fc(output).squeeze()
        output = self.sig(output)
        
        return output

class PolarityLSTM(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_dim, output_size, n_layers, drop_lstm=0.1, drop_out = 0.1):
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_lstm, batch_first=True)
        self.dropout = nn.Dropout(drop_out)
        # self.fc = nn.Linear(hidden_dim, int(hidden_dim/2))
        # self.sig = nn.ReLU()
        # self.fc2 = nn.Linear(int(hidden_dim/2), output_size)
        # self.sig2 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_size)
        self.sig2 = nn.Sigmoid()

    def forward(self, x, seq_lengths):  
        embedded_seq_tensor = self.embedding(x)
        packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)

        packed_output, (ht, ct) = self.lstm(packed_input, None)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
       
        last_idxs = (input_sizes - 1).to(device) 
        output = torch.gather(output, 1, last_idxs.view(-1, 1).unsqueeze(2).repeat(1, 1, self.hidden_dim)).squeeze() 

        output = self.dropout(output)
        # output = self.fc(output).squeeze()
        # output = self.sig(output)
        output = self.fc2(output).squeeze()
        output = self.sig2(output)
        
        return output

def criterion(out, label):
    return functional.binary_cross_entropy(out, label)

def test(test_data, labels):
    # with open(conf.readValue("lstm_model_path"), "rb") as file:
    #     model = pickle.load(file)
    if("-gru" in sys.argv):
        model = PolarityGRU(embedding_dim, vocab_size, hidden_dim, output_size, n_layers)
    else:
        model = PolarityLSTM(embedding_dim, vocab_size, hidden_dim, output_size, n_layers)
    
    model.load_state_dict(torch.load(conf.readValue("lstm_model_path")))


    if("-gpu" in sys.argv):
        model.cuda(device)


    pb = ProgressBar(total=int(len(test_data['embedding'])-1/batch_size),prefix='Training in progress', suffix='', decimals=3, length=50, fill='X', zfill='-')

    test_generator = DataSampler(seq_tensor, seq_lengths, labels, batch_size)
    model.eval()
    evaluator = Evaluator()

    outputs = []
    labels = []
    counter = 0

    LOGGER.debug("Evaluation in progress")

    for subset_input_tensor, subset_input_lengths, subset_labels_tensor in iter(test_generator):
        pb.print_progress_bar(counter)

        subset_input_tensor = subset_input_tensor.to(device)
        subset_input_lengths = subset_input_lengths.to(device)
        subset_labels_tensor = subset_labels_tensor.to(device)

        if("-gpu" in sys.argv and "-gru" in sys.argv):
            model.gru.flatten_parameters()
        elif("-gpu" in sys.argv):
            model.lstm.flatten_parameters()

        try:
            output = model(subset_input_tensor, subset_input_lengths)
        except RuntimeError as ex:
            print(counter)
            print(ex)
            print(subset_input_tensor)
            print(subset_input_lengths)
            continue
        
        binary_output = (output >= 0.5).short()
        outputs.extend(binary_output.cpu().detach().numpy())
        labels.extend(subset_labels_tensor.cpu().detach().numpy())    
        counter += 1
    
    return evaluator.evaluate(labels, outputs)



if __name__ == "__main__":
    if "--log" in sys.argv:
            logging.basicConfig(level=logging.DEBUG)
    conf = Config()
    with open(conf.readValue("vocabulary"), "rb") as f:
        vocabulary = pickle.load(f)

    chunk_size = 100000
    
    vocab_size = vocabulary.getVocabLength()

    accuracy_array = []
    fscore_array = []
    precision_array = []
    recall_array = []
    test_accuracy_array = []
    loss_array = []
    time_array = []
    start_time = time.time()
    if("-train" in sys.argv):
        if("-gru" in sys.argv):
            LOGGER.debug("training GRU model")
            model = PolarityGRU(embedding_dim, vocab_size, hidden_dim, output_size, n_layers)
        else:
            LOGGER.debug("training LSTM model")
            model = PolarityLSTM(embedding_dim, vocab_size, hidden_dim, output_size, n_layers)
        if("-gpu" in sys.argv):
            model.cuda(device)

    for dupa in range(4,7):

        LOGGER.debug("Reading data")
        if("-train" in sys.argv):
            if(dupa != 0):
                data = pd.read_csv(conf.readValue("processed_data_set"), sep=";", skiprows=int(dupa*chunk_size), nrows = chunk_size, names=['id', 'embedding', 'polarity'] )
            else:
                data = pd.read_csv(conf.readValue("processed_data_set"), sep=";", skiprows=int(dupa*chunk_size), nrows = chunk_size)
            # data = data[:set_count]
            
            data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        LOGGER.debug("offset: " + str(dupa*chunk_size))
        # elif("-test" in sys.argv):
        test_data = pd.read_csv(conf.readValue("processed_test_set"), sep=";")
        test_data = test_data[:int(0.3*set_count)]
        test_data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

        vocab_size = vocabulary.getVocabLength()
        vocab_to_int = vocabulary.getVocab2int()
        vectorized_seqs = []

        # if("-train" in sys.argv or "-test" in sys.argv):

        LOGGER.debug("Vectorization and tokenization")
        for seq in data['embedding']:
            if isinstance(seq, str): 
                vectorized_seqs.append([vocab_to_int.get(word,1) for word in TOKENIZER.tokenize(seq)])
            else:
                vectorized_seqs.append([])

        seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))
        # labels = torch.LongTensor(list(map(lambda x: 1 if x == 'positive' else 0, data['polarity'])))
        labels = torch.LongTensor(data['polarity'])

        LOGGER.debug("Adding padding")
        seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long()
        for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

        LOGGER.debug("Model created")



        """
            Params end
        """
        accuracy_array = []
        fscore_array = []
        precision_array = []
        recall_array = []
        test_accuracy_array = []
        loss_array = []
        time_array = []
        start_time = time.time()
        if("-train" in sys.argv):
            # if("-gru" in sys.argv):
            #     LOGGER.debug("training GRU model")
            #     model = PolarityGRU(embedding_dim, vocab_size, hidden_dim, output_size, n_layers)
            # else:
            #     LOGGER.debug("training LSTM model")
            #     model = PolarityLSTM(embedding_dim, vocab_size, hidden_dim, output_size, n_layers)
            # if("-gpu" in sys.argv):
            #     model.cuda(device)
            generator = DataSampler(seq_tensor, seq_lengths, labels, batch_size)
            
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, len(data['embedding']), eta_min=learning_rate)
            LOGGER.debug("Training in progress")
            LOGGER.debug("Training on set of size: {}".format(len(data['embedding'])))
            pb = ProgressBar(total=int(len(data['embedding'])-1/batch_size),prefix='Training in progress', suffix='', decimals=3, length=50, fill='X', zfill='-')
            model.train()
            for e in range(epochs):
                LOGGER.debug("Epoch {}/{}".format(e, epochs))
                counter = 0
                correct = []
                total = []
                for subset_input_tensor, subset_input_lengths, subset_labels_tensor in iter(generator):
                    # print(subset_input_tensor, subset_input_lengths, subset_labels_tensor)

                    subset_input_tensor_tmp, subset_input_lengths_tmp, subset_labels_tensor_tmp = subset_input_tensor, subset_input_lengths, subset_labels_tensor
                    pb.print_progress_bar(counter)
                    counter += 1
                        
                    try:
                        subset_input_tensor = subset_input_tensor.to(device)
                        subset_input_lengths = subset_input_lengths.to(device)
                        subset_labels_tensor = subset_labels_tensor.to(device)
            
                    
                        output = model(subset_input_tensor, subset_input_lengths)
                    except RuntimeError as ex:
                        print(counter)
                        print(ex)
                        print(subset_input_tensor_tmp)
                        print(subset_input_lengths_tmp)
                        print(subset_labels_tensor_tmp)
                        model.cuda(device)
                        continue
                        
                    loss = criterion(output, subset_labels_tensor.float())
                    # return
                    optimizer.zero_grad() 
                    loss.backward()
                    
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()

                    # Calculate accuracy

                    binary_output = (output >= 0.5).short()
                    right_or_not = torch.eq(binary_output, subset_labels_tensor)
                    correct.append(torch.sum(right_or_not).float().item())
                    total.append(right_or_not.shape[0])
                
                scheduler.step(e)
                # LOGGER.debug(binary_output)
                # LOGGER.debug(subset_labels_tensor)
                accuracy = sum(correct) / sum(total)
                correct.clear()
                total.clear()
                LOGGER.debug("Loss function: {:2f}, accuracy: {:3f}".format(loss, accuracy))
                LOGGER.debug("Steps taken: {}".format(counter))

                torch.save(model.state_dict(), conf.readValue("lstm_model_path"))
                metrics = test(test_data, labels)

                # model.train()

                accuracy_array.append(accuracy)
                loss_array.append(loss.item())
                fscore_array.append(metrics['f-score'])
                precision_array.append(metrics['precision'])
                recall_array.append(metrics['recall'])
                test_accuracy_array.append(metrics['accuracy'])
                time_array.append(time.time() - start_time)
                start_time = time.time()

            LOGGER.debug("Training finished")
            # f = open('./accuracy_train_epochs.txt', 'w')
            # for i, a in enumerate(accuracy_array):
            #     f.write(str(i) + " " + str(a) + "\n" )
            # f.close()

            d = {'Epoch' : range(1,epochs +1), 'Accuracy' : accuracy_array, 'f-score' : fscore_array, 'Loss' : loss_array,
                    'Precision' : precision_array, 'Recall': recall_array, 'Test set accuracy': test_accuracy_array, 'Learning time': time_array}
            df = pd.DataFrame(d,columns=['Epoch','Accuracy', 'f-score', 'Precision', 'Recall', 'Test set accuracy', 'Loss', 'Learning time'])
            df.to_csv('./metrics_epochs.csv', sep = ';')
            ax = plt.gca()
            df.plot(x ='Epoch', y='Accuracy', kind = 'line', color='red', ax=ax)
            df.plot(x ='Epoch', y='f-score', kind = 'line', color='green', ax=ax)
            df.plot(x ='Epoch', y='Precision', kind = 'line', color='blue', ax=ax)
            df.plot(x ='Epoch', y='Recall', kind = 'line', color='yellow', ax=ax)
            df.plot(x ='Epoch', y='Test set accuracy', kind = 'line', color='purple', ax=ax)
            df.plot(x ='Epoch', y='Loss', kind = 'line', color='brown', ax=ax)
            plt.savefig('./accuracy_train_epochs_' + str(dupa) + '.png')



            # with open(conf.readValue("lstm_model_path"), "wb") as file:
            #     pickle.dump(model, file)
            # ts = time.time()
            torch.save(model.state_dict(), conf.readValue("lstm_model_path"))
            LOGGER.debug("Model serialized")
        
    # if("-test" in sys.argv):
    #     test()
        # with open(conf.readValue("lstm_model_path"), "rb") as file:
        #     model = pickle.load(file)
        # model = PolarityLSTM(embedding_dim, vocab_size, hidden_dim, output_size, n_layers)
        # model.load_state_dict(torch.load(conf.readValue("lstm_model_path")))


        # if("-gpu" in sys.argv):
        #     model.cuda(device)

        # pb = ProgressBar(total=int(len(data['embedding'])-1/batch_size),prefix='Training in progress', suffix='', decimals=3, length=50, fill='X', zfill='-')

        # test_generator = DataSampler(seq_tensor, seq_lengths, labels, batch_size)
        # model.eval()
        # evaluator = Evaluator()

        # outputs = []
        # labels = []
        # couter = 0

        # LOGGER.debug("Evaluation in progress")

        # for subset_input_tensor, subset_input_lengths, subset_labels_tensor in iter(test_generator):
        #     pb.print_progress_bar(counter)

        #     subset_input_tensor = subset_input_tensor.to(device)
        #     subset_input_lengths = subset_input_lengths.to(device)
        #     subset_labels_tensor = subset_labels_tensor.to(device)

        #     if("-gpu" in sys.argv):
        #         model.lstm.flatten_parameters()

        #     try:
        #         output = model(subset_input_tensor, subset_input_lengths)
        #     except RuntimeError as ex:
        #         print(counter)
        #         print(ex)
        #         print(subset_input_tensor)
        #         print(subset_input_lengths)
        #         continue
            
        #     binary_output = (output >= 0.5).short()
        #     outputs.extend(binary_output.cpu().detach().numpy())
        #     labels.extend(subset_labels_tensor.cpu().detach().numpy())    
        #     counter += 1
        
        # evaluator.evaluate(labels, outputs)

    
    if("-predict" in sys.argv):
        # with open(conf.readValue("lstm_model_path"), "rb") as file:
        #     model = pickle.load(file)
        model = PolarityLSTM(embedding_dim, vocab_size, hidden_dim, output_size, n_layers)
        model.load_state_dict(torch.load(conf.readValue("lstm_model_path")))

        model.eval()
        if("-gpu" in sys.argv):
            model.cuda(device)
        prep = Preprocessor()

        index = sys.argv.index("-predict")

        text = sys.argv[index+1]
        text = prep.setText(text).correctSpelling().setLemmatizeFlag().setStopWordsFlag().build()
        text = [text]

        vectorized_seqs = []
        for seq in text: 
            vectorized_seqs.append([vocab_to_int.get(word,1) for word in TOKENIZER.tokenize(seq)])
        
        seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))

        seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long()
        for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

        # print(seq_tensor)
        # print(seq_lengths)

            seq_tensor = seq_tensor.to(device)
            seq_lengths = seq_lengths.to(device)
            if("-gpu" in sys.argv):
                    model.lstm.flatten_parameters()
            try:
                output = model(seq_tensor, seq_lengths)
            except RuntimeError as ex:
                print(counter)
                print(ex)
                print(seq_tensor)
                print(seq_lengths)
                continue

        value = output.item()
        label = 'positive' if value >= 0.5 else 'negative'
        print('Polarity of given sentence is ' + label + ", and exquals to: {}".format(value))

"""
    Usage:
        -train to train lstm model
        -test to evaluate model
        -predict to predict sentence given as 2nd argument
        --log to print logs
        -gpu to enable gpu support (if available)
        -gru to train using gru model
"""

"""
    Jeśli długość była dobrym predyktorem dla baseline'a a lstm sobie nie radził to może dodąć długość do warstwy w pełni połączonej?
"""
