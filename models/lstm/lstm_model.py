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

from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.config import Config
from utils.process_results import Vocabulary
from nltk.tokenize import RegexpTokenizer
from console_progressbar import ProgressBar
from utils.data_evaluator import Evaluator

TOKENIZER = RegexpTokenizer(r'\w+')
LOGGER = logging.getLogger('lstm_model')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataSampler(object):
    
    def __init__(self, input_tensor, input_lengths, labels_tensor, batch_size, sequence_lenght = 2665):
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

def criterion(out, label):
    return functional.binary_cross_entropy(out, label)


if __name__ == "__main__":
    if "--log" in sys.argv:
            logging.basicConfig(level=logging.DEBUG)
    conf = Config()
    with open(conf.readValue("vocabulary"), "rb") as f:
        vocabulary = pickle.load(f)

    LOGGER.debug("Reading data")
    if("-train" in sys.argv):
        data = pd.read_csv(conf.readValue("processed_data_set"), sep=";")
    elif("-test" in sys.argv):
        data = pd.read_csv(conf.readValue("processed_test_set"), sep=";")

    vocab_to_int = vocabulary.getVocab2int()
    vectorized_seqs = []

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
        Params start
    """

    epochs = 10
    counter = 0
    learning_rate = 0.0001
    weight_decay = 0.005
    momentum = 0.9
    clip = 5
    embedding_dim = 300
    vocab_size = vocabulary.getVocabLength()
    hidden_dim = 300
    output_size = 1
    n_layers = 2
    batch_size = 80

    """
        Params end
    """
    
    if("-train" in sys.argv):
        model = PolarityLSTM(embedding_dim, vocab_size, hidden_dim, output_size, n_layers)
        generator = DataSampler(seq_tensor, seq_lengths, labels, batch_size)
        
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
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
                pb.print_progress_bar(counter)
                counter += 1
                    
                subset_input_tensor = subset_input_tensor.to(device)
                subset_input_lengths = subset_input_lengths.to(device)
                subset_labels_tensor = subset_labels_tensor.to(device)
        
                output = model(subset_input_tensor, subset_input_lengths)
            
                loss = criterion(output, subset_labels_tensor.float())
                
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

        LOGGER.debug("Training finished")

        with open(conf.readValue("lstm_model_path"), "wb") as file:
            pickle.dump(model, file)
        LOGGER.debug("Model serialized")
        
    if("-test" in sys.argv):
        with open(conf.readValue("lstm_model_path"), "rb") as file:
            model = pickle.load(file)

        pb = ProgressBar(total=int(len(data['embedding'])-1/batch_size),prefix='Training in progress', suffix='', decimals=3, length=50, fill='X', zfill='-')

        test_generator = DataSampler(seq_tensor, seq_lengths, labels, batch_size)
        model.eval()
        evaluator = Evaluator()

        outputs = []
        labels = []
        couter = 0

        LOGGER.debug("Evaluation in progress")

        for subset_input_tensor, subset_input_lengths, subset_labels_tensor in iter(test_generator):
            pb.print_progress_bar(counter)

            subset_input_tensor = subset_input_tensor.to(device)
            subset_input_lengths = subset_input_lengths.to(device)
            subset_labels_tensor = subset_labels_tensor.to(device)
            output = model(subset_input_tensor, subset_input_lengths)

            binary_output = (output >= 0.5).short()
            outputs.extend(binary_output.detach().numpy())
            labels.extend(subset_labels_tensor.detach().numpy())    
            counter += 1
        
        evaluator.evaluate(labels, outputs)