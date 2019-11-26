from utils.data_reader import DataReader
from utils.data_evaluator import Evaluator
import multiprocessing as mp
import fasttext as ft
import csv, os

FOLDER = "fasttext_tool/"
def saveInfoToFile(row, output):
    output.write("__label__{} {}\n".format(row['polarity'], str(row['text'])))
    return ""

def adjustForm(dataSet, fileName):
    print("Transforming...")
    with open('{}{}'.format(FOLDER, fileName), 'w+') as output:
        dataSet.apply(lambda x: saveInfoToFile(x, output), axis=1)

if __name__ == "__main__":
    dataReader = DataReader()
    evaluator = Evaluator()
    if not "data.train" in os.listdir(FOLDER):
        dataSet = dataReader.read_data_set()
        adjustForm(dataSet, "data.train")
    if not "data.test" in os.listdir(FOLDER):
        testSet = dataReader.read_test_set()
        adjustForm(testSet, "data.test")
    if not "model.bin" in os.listdir(FOLDER):
        model = ft.train_supervised(input=FOLDER + "data.test")
        model.save_model(FOLDER + "model.bin")
    else:
        model = ft.load_model(FOLDER + "model.bin")
    (_, precision, recall) = model.test(FOLDER + "data.test")
    metrics = {'precision': precision, 'recall': recall, 'fscore': evaluator.calculate_fscore(precision, recall)}
    evaluator.print(metrics)
