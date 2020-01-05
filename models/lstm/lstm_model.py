import torch.utils.data.sampler as splr
import numpy as np
import pandas as pd
import logging
import sys
import pickle
import torch.nn as nn
import torch

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.config import Config
from utils.process_results import Vocabulary
from nltk.tokenize import RegexpTokenizer

TOKENIZER = RegexpTokenizer(r'\w+')
LOGGER = logging.getLogger('lstm_model')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataSampler(object):
    
    def __init__(self, input_tensor, input_lengths, labels_tensor, batch_size, sequence_lenght = 2665):
        if "--log" in sys.argv:
            logging.basicConfig(level=logging.DEBUG)
        self.input_tensor = input_tensor
        self.input_lengths = input_lengths
        self.labels_tensor = labels_tensor
        self.batch_size = batch_size
        self.sequence_lenght = 2665

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
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, seq_lengths):  
        embedded_seq_tensor = self.embedding(x)
        packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)

        packed_output, (ht, ct) = self.lstm(packed_input, None)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
       
        last_idxs = (input_sizes - 1).to(device) 
        output = torch.gather(output, 1, last_idxs.view(-1, 1).unsqueeze(2).repeat(1, 1, self.hidden_dim)).squeeze() 

        output = self.dropout(output)
        output = self.fc(output).squeeze()
        output = self.sig(output)
        
        return output

if __name__ == "__main__":
    conf = Config()
    with open(conf.readValue("vocabulary"), "rb") as f:
        vocabulary = pickle.load(f)

    data = pd.read_csv(conf.readValue("processed_data_set"), sep=";")

    vocab_to_int = vocabulary.getVocab2int()
    vectorized_seqs = []

    for seq in data['embedding']:
        if isinstance(seq, str): 
            vectorized_seqs.append([vocab_to_int.get(word,1) for word in TOKENIZER.tokenize(seq)])
        else:
            vectorized_seqs.append([])

    seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))
    # labels = torch.LongTensor(list(map(lambda x: 1 if x == 'positive' else 0, data['polarity'])))
    labels = torch.LongTensor(data['polarity'])

    seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long()
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

    model = PolarityLSTM(300, vocabulary.getVocabLength(), 300, 1, 2)
    generator = DataSampler(seq_tensor, seq_lengths, labels, 80)

    for subset_input_tensor, subset_input_lengths, subset_labels_tensor in iter(generator):
        output = model.forward(subset_input_tensor, subset_input_lengths)
        print(output)
        break