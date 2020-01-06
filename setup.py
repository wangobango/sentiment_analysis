import os
import subprocess
from utils.preprocessor import Preprocessor
from utils.test_set_splitter import main

DATA_FOLDER="data"
AGGREGATED="aggregated"
DATA_SET = "data_set"
TEST_SET = "test_set"
if __name__ == "__main__":
    listdir = os.listdir()
    if not DATA_FOLDER in listdir:
        subprocess.run(["./download_dataset.sh"])
    if not AGGREGATED in listdir:
        os.mkdir(AGGREGATED)
        Preprocessor.aggregateData()
    if not DATA_SET in listdir or not TEST_SET in listdir:
        main()
