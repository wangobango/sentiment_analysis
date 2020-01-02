from utils.config import Config
from .data_preparator import LABEL_MAP
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer
import pickle, copy
import sys,csv

class InputExample():
    def __init__(self, guid, text_a, text_b = None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class BertDataConverter():
    def __init__(self):
        self.config = Config()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    def get_dev_examples(self):
        dev_set = self.read_tsv(self.config.readValue('bert_dev_set'), "dev")
        return dev_set

    @staticmethod
    def read_tsv(input_file, set_type,quotechar=None):
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for i,line in enumerate(reader):
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                guid = "%s-%s" % (set_type, i)
                text_a = line[3]
                label = line[1]
                lines.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            return lines

    def convert_example_to_features(self, example):
        max_seq_length = 512
        tokenizer = self.tokenizer
        tokens = tokenizer.tokenize(example.text_a)
        if len(tokens) > max_seq_length - 2: # -2 for [CLS] and [SEP]
            tokens = tokens[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(tokens)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_id = example.label
        return InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id)

def convert_examples_to_features(bertDataConverter, examples):
    process_count = cpu_count() - 1
    with Pool(process_count) as p:
        train_features = \
            list(tqdm(p.imap(bertDataConverter.convert_example_to_features, examples), total=len(examples)))
    with open('./huggingfaces' + 'train_features.pkl', "wb") as f:
        pickle.dump(train_features, f)

if __name__ == "__main__":
    bertDataConverter = BertDataConverter()
    examples = bertDataConverter.get_dev_examples()
    convert_examples_to_features(bertDataConverter, examples)


#MAX sequence length
#LABEL MAP
#tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)