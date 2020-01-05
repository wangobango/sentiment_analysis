import pickle, logging, torch, pickle
from .data_converter import InputFeatures
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

PREFIX = "./huggingfaces/"
BERT_MODEL = 'bert-base-cased'
CACHE_DIR = PREFIX + 'cache/'
TRAIN_BATCH_SIZE = 24
NUM_TRAIN_EPOCHS = 1
OUTPUT_MODE = 'classification'
WEIGHTS_NAME = "pytorch_model.bin"
NUM_LABEL=2
LEARNING_RATE=2e-5
WARMUP_PROPORTION = 0.1
GRADIENT_ACCUMULATION_STEPS = 1
LOGGER = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    with open('./huggingfaces/train_features.pkl', 'rb') as input:
        inputFeatures = pickle.load(input)
    train_examples_len = len(inputFeatures)
    num_train_optimization_steps = int(train_examples_len / TRAIN_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS) * NUM_TRAIN_EPOCHS

    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, cache_dir=CACHE_DIR, num_labels=NUM_LABEL)
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                    lr=LEARNING_RATE,
                    warmup=WARMUP_PROPORTION,
                    t_total=num_train_optimization_steps)
    
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE)

