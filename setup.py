import os
import subprocess

DATA_FOLDER="data"
if __name__ == "__main__":
    if not DATA_FOLDER in os.listdir():
        subprocess.run(["./download_dataset.sh"])
    