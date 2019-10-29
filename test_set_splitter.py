import pandas as pd
from config import Config
import os
import glob

PATH = "aggregated_path"
TEST_SET_PATH = "test_set_path"
DATA_SET_PATH = "data_set_path"
TEST_SET_FRACTION = "test_set_fraction"

class Test_set_splitter:
    ###
    # If called with no arguments, will use the defaults from config.json
    ### Example:
    # splitter = Test_set_splitter(input_data_folder_path="test")
    def __init__(self, input_data_folder_path = None, output_test_set_path = None, output_data_set_path = None, fraction_of_data_to_move_to_test_set = None):
        config = Config()

        self.data_folder_path = input_data_folder_path if input_data_folder_path is not None else config.readValue(PATH)
        self.test_set_path = output_test_set_path if output_test_set_path is not None else config.readValue(TEST_SET_PATH)
        self.data_set_path = output_data_set_path if output_data_set_path is not None else config.readValue(DATA_SET_PATH)
        self.test_set_fraction = fraction_of_data_to_move_to_test_set if fraction_of_data_to_move_to_test_set is not None else config.readValue(TEST_SET_FRACTION)

    ###
    # internal
    def split(self, file_path):
        dataFrame = pd.read_csv(file_path)
        dataFrame.drop(dataFrame.columns[0], axis=1, inplace=True)
        nrows = dataFrame.shape[0]

        n_test_set_rows = int(nrows * self.test_set_fraction)

        return dataFrame[:n_test_set_rows], dataFrame[n_test_set_rows:]
    
    ###
    # splits all files in self.data_folder_path to test_set and data_set
    def splitAll(self, log = False):
        complete_path = os.path.join(self.data_folder_path, "*.csv")
        all_files = glob.glob(complete_path)

        if not os.path.exists(os.path.dirname(self.test_set_path)):
            os.makedirs(os.path.dirname(self.test_set_path))

        if not os.path.exists(os.path.dirname(self.data_set_path)):
            os.makedirs(os.path.dirname(self.data_set_path))


        all_files_len = len(all_files)
        for index, file_path in enumerate(all_files):
            print("Splitting ", file_path, ". Progress is ", index+1, "/", all_files_len)
            test_set_part, data_set_part = self.split(file_path)
            
            if (index == 0):
                test_set = test_set_part
                data_set = data_set_part
            else:
                test_set = pd.concat([test_set, test_set_part])
                data_set = pd.concat([data_set, data_set_part])
            
            if (log):
                print("test_set_part size =", test_set_part.shape[0], "data_set_part size =", data_set_part.shape[0])
                print("test_set size =", test_set.shape[0], "data_set size =", data_set.shape[0], "\n")

        print("Saving to files {}, {}".format(self.test_set_path, self.data_set_path))
        test_set.to_csv(self.test_set_path, index = False)
        data_set.to_csv(self.data_set_path, index = False)
    
    ###
    # use this to append to exising test_set and data_set
    ### Example:
    # splitter = Test_set_splitter()
    # splitter.splitAndAppendToExisting("./test/Amazon_Instant_Video.csv")
    def splitAndAppendToExisting(self, file_path):
        test_set_part, data_set_part = self.split(file_path)

        test_set = pd.read_csv(self.test_set_path)
        # test_set.drop(test_set.columns[0], axis=1, inplace=True)
        data_set = pd.read_csv(self.data_set_path)
        # data_set.drop(data_set.columns[0], axis=1, inplace=True)

        print("Appending {} to existing test_set and data_set\nBEFORE:\ntest_set size is {}, data_set size is {}".format(file_path, test_set.shape[0], data_set.shape[0]))

        test_set = pd.concat([test_set, test_set_part])
        data_set = pd.concat([data_set, data_set_part])

        print("AFTER:\ntest_set size is {}, data_set size is {}".format(test_set.shape[0], data_set.shape[0]))

        test_set.to_csv(self.test_set_path, index = False)
        data_set.to_csv(self.data_set_path, index = False)

if __name__ == "__main__":
    splitter = Test_set_splitter()
    splitter.splitAll()