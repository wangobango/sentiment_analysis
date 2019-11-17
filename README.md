# Bachelor's thesis involving sentiment analysis.

Our project consists of 4 people etc etc.. Marcel will finish the rest of it anyway...

# Prerequisites

Type the following command to install required packages :

```
sudo pip3 install -r requirements.txt
```
Create following directories to store postproducts of scripts:
```
mkdir plots
```
Download the required data sets, unzip them and store the results in ```data``` directory.

# Test_set_splitter
Component whose job is to split the data into test set and data set. The test set will later be used to validate and rank the analysis approaches used.

It is mandatory to first use:
```
python3 -m utils.data_exploration
```

To aggregate data in ```aggregated``` directory.

## Usage
The component consists of a class Test_set_splitter and it's public methods:
`SplitAll()` and `SplitAndAppendToExisting()`

To use them, the data needs to be in .csv format in folder `aggregated`. The component will then split the data into test_set/test_set.csv and data_set/data_set.csv

To split all data use:
```
python3 -m utils.test_set_splitter
```

To get more verbose information use:
```
python3 -m utils.test_set_splitter --log
```

To append new data to existing sets use:
```
python3 -m utils.test_set_splitter --append [path_to_file]
```

# Data exploration

## Data exploration script provides following functions:
  * Aggregating data in ```data``` folder, which consits of multiple .xml files seperated in each domain. Script then saves data to one .csv file for each domain to ```aggregated``` directory

  * Preparing ```results.json``` file for each domain. File contains results of premilimnary analysis of data sets. Examples of few of the attributes exported to json file: 
    - numberOfPositives
    - numberOfNegatives
    - averageTextLengthWhenPolarityPositiveChars
    - averageTextLengthWhenPolarityNegativeChars 
  etc ...
  
  * Prepering plots of selected attributes to visualise the data.

## Script can be be started with following arguments: 
  * ```-debug``` - to print error logs
  * ```-dump``` - to dump the analysis results to results.json file
  * ```-plot``` - to show plot analysis as well as save plots to ```plot``` directory
  * ```-aggregate``` - to aggregate data in ```aggregated``` directory

## Known issues:
  * 1# at this point ```results``` analysis sometimes crashes. This exception is then caught and ```21.37``` value is saved to results temoporarly

# Baseline models

## Mean length of text based model
This model uses linear regression, as well as SGD to calculate loss function. 
### Prerequisites
  - have generated ```test_set``` and ```data_set```

### Usage
To teach model and save it to binary file for later use, type :
```
python3 -m baselines.mean_length_baseline -teach
```
To evaluate the model on a test set:
```
python3 -m baselines.mean_length_baseline -evaluate
```

## Counting text score based on words' polarity
This model calulates border value based on text score which later is used to decide if text has 'positive' or 'negative' polarity. Border value is just mean of all text scores from ```data_set```
### Prerequisites
  - have generated ```test_set``` and ```data_set```
### Usage
To teach model and save it to binary file for later use, type :
```
python3 -m baselines.word_polarity_counting -teach
```
To evaluate the model on a test set:
```
python3 -m baselines.word_polarity_counting -evaluate
```
Aditional options:
```
--log - to get more information about script process
```
```
--parallel - to tech model on multiple processors, by default it uses only one :(
```
```
--help - to get more information
```