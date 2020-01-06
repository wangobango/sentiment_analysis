from .config import Config
import pandas as pd
from random import seed
from random import randint
import numpy as np
import pickle


class Batch:

    def __init__(self, size, ceiling):
        self.size = size
        self.ceiling = ceiling
        self.sequences = []
        self.sequences_s = None

    def can_fit(self, length):
        if(len(self.sequences) < self.size and length <= self.ceiling):
            return True
        else: 
            return False

    def add_sequence(self, df):
        self.sequences.append(df)

    def add_padding(self):
        new_sequences = []
        tmp = []
        for s in self.sequences:
            for i in range(self.ceiling - s.shape[0]):
                tmp.append({"id": s.iat[0,0],
                            "embedding" : np.zeros(300),
                            "polarity" : s.iat[0,2]})
            
            if len(tmp) > 0:
                tmp = pd.DataFrame(tmp)
                new_s = pd.concat([s, tmp])
                new_sequences.append(new_s)
            else:
                new_sequences.append(s)
            tmp = []

        self.sequences = new_sequences

    def prepare_sequences(self):
        self.sequences_s = pd.concat(self.sequences)


class Paddinger:

    def __init__(self, batch_count, file_name = 0):
        self.batch_count = batch_count
        self.sequences = 0
        self.config = Config()
        self.batches = []
        self.file_name = file_name

    def tmp_add_ids(self):
        if self.file_name != 0:
            data_set = pd.read_csv(self.file_name, sep = ';',  index_col=0)
        else:
            data_set = pd.read_csv(self.config.readValue('processed_data_set'), sep = ';',  index_col=0)
        # data_set.rename(columns = {'Unnamed: 0' : 'relative_id'})
        data_set.sort_values(by = ['id'], inplace = True, kind = 'mergesort')
        return data_set

    def get_padding_stats(self, data):
        indexes = data['id']
        arr_of_len = dict()
        curr_id = data.iat[0,0]
        length = 0
        for i,(id, row) in enumerate(data.iterrows()):
            if row['id'] == curr_id and i != len(data) -1:
                length +=1
            else:
                curr_id = row['id']
                self.sequences +=1
                if length in arr_of_len:
                    arr_of_len[length] +=1
                else:
                    arr_of_len[length] = 1

                length = 1

        return arr_of_len
        # return sorted(arr_of_len.keys())
    
    def get_batches(self):
        data = self.tmp_add_ids()
        stats = self.get_padding_stats(data)
        size_of_batch= int(self.sequences / self.batch_count)
        intervals = []
        ceiling = 0
        curr_size = 0
        lengths_as_array = []
        for length in sorted(stats):
            count = stats[length]
            for i in range(count):
                lengths_as_array.append(length)
        self.init_batches(lengths_as_array, size_of_batch)
        curr_id = data.iat[0,0]
        length = 0
        tmp = len(data) -1
        tmp_df = []

        for i,(id, row) in enumerate(data.iterrows()):
            if row['id'] == curr_id and i != tmp:
                length +=1
                tmp_df.append(row)
            else:
                for batch in self.batches:
                    if batch.can_fit(length):
                        batch.add_sequence(pd.DataFrame(tmp_df))
                        tmp_df = [row]
                        curr_id = row['id']
                        length = 1
                        break
        
        self.padding()
        for b in self.batches:
            b.prepare_sequences()

        return self.batches


    

    def init_batches(self, lengths, size_of_batch):
        # print(len(lengths))
        for i in range(1 , self.batch_count +1):
            ceiling = lengths[(i * size_of_batch) - 1]
            batch = Batch(size_of_batch, ceiling)
            self.batches.append(batch)
    

    def padding(self):
        for b in self.batches:
            b.add_padding()

    def serialize(self, s_type, name):
        if(s_type == 'pickl'):
            print('to pickl')
            out = open(name, 'wb')
            pickle.dump(self.batches, out)
            out.close()
            print('done')
        elif (s_type == 'csv'):
            print('to csv')
            

        else:
            print('wrong serialize type')





        # for i in range (1, paddinger.batch_size +1):
            


#    example of usage
#    Class creates a list of batches with the same sequence length. 
#    @Args:
#       batch_count : mandatory. How many batches you want to have?
#       file_name : optional. Path to pocessed dataset. If not set, takes the one from config
# if __name__ == "__main__":
#     paddinger = Paddinger(batch_count = 7)
#     batches = paddinger.get_batches()
    # for b in batches:
    #     print(b.ceiling)
    #     print(b.sequences_s)  #Here all sequences are in one dataframe
# 
#     or
# 
    # for b in batches:
    #     print(b.ceiling)
    #     for seq in b.sequences:
    #         print(seq.shape) #Here sequences are represented as array of Dataframes of particular sequences


    
    
