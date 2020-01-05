
import pandas as pd
import logging, pickle, sys
from simpletransformers.classification import ClassificationModel
from utils.data_reader import DataReader

LOGGER = logging.getLogger("SimpleTranformers")
LABEL_MAP = {'negative': int(0), 'positive': int(1)}
MODEL_PATH = "./huggingfaces/model.pkl"

def readSets(amount=None):
    dataReader = DataReader()
    train_df = dataReader.read_data_set()
    if amount is not None:
        train_df = train_df[:amount]
    eval_df = dataReader.read_test_set()
    if amount is not None:
        eval_df = eval_df[:amount]
    return train_df, eval_df

def adjustSets(train_df, eval_df):
    train_df = pd.DataFrame({
        'text': train_df['text'].replace(r'\n', ' ', regex=True),
        'label':train_df['polarity'].replace(LABEL_MAP)
    })
    eval_df = pd.DataFrame({
        'text': eval_df['text'].replace(r'\n', ' ', regex=True),
        'label':eval_df['polarity'].replace(LABEL_MAP)
    })
    return train_df, eval_df

if __name__ == "__main__":
    train_df, eval_df = readSets(100)
    train_df, eval_df = adjustSets(train_df, eval_df)
    print(train_df.head())
    print(eval_df.head())
    # Create a TransformerModel
    if "--force" in sys.argv:
        model = ClassificationModel('bert', 'bert-base-uncased', use_cuda=False, args={'overwrite_output_dir': True})
        \
            LOGGER.info("Training model...")
        model.train_model(train_df)
        with open(MODEL_PATH, 'wb') as output:
            pickle.dump(model, output)
    else:
        with open(MODEL_PATH, 'rb') as input:
            model = pickle.load(input)
    \
        LOGGER.info("Evaluating model...")
    print(model.predict(eval_df['text']))f
    # result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    # print(result, model_outputs, wrong_predictions)