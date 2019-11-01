# Bachelor's thesis involving sentiment analysis.

Our project consists of 4 people etc etc.. Marcel will finish the rest of it anyway...

## Test_set_splitter
Component whose job is to split the data into test set and data set. The test set will later be used to validate and rank the analysis approaches used.

### Usage
The component consists of a class Test_set_splitter and it's public methods:
`SplitAll()` and `SplitAndAppendToExisting()`

To use them, the data needs to be in .csv format in folder `aggregated`. The component will then split the data into test_set/test_set.csv and data_set/data_set.csv

To split all data use:
```
python3 test_set_splitter
```

To get more verbose information use:
```
python3 test_set_splitter --log
```

To append new data to existing sets use:
```
python3 test_set_splitter --append [path_to_file]
```
