import os
import subprocess
from utils.preprocessor import Preprocessor

DATA_FOLDER="data"
AGGREGATED="aggregated"

if __name__ == "__main__":
    if not DATA_FOLDER in os.listdir():
        subprocess.run(["./download_dataset.sh"])
    if not AGGREGATED in os.listdir():
        os.mkdir(AGGREGATED)
        Preprocessor.aggregateData()
