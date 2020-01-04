import pickle
from .data_converter import InputFeatures

with open('./huggingfacestrain_features.pkl', 'rb') as input:
    x = pickle.load(input)
    print(x)