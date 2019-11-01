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

## Known issues:
  * 1# at this point ```results``` analysis sometimes crashes. This exception is then caught and ```21.37``` value is saved to results temoporarly