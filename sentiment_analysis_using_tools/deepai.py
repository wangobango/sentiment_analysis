### Using https://deepai.org/machine-learning-model/sentiment-analysis sentiment analysis tool

# Example of using API:
# import requests
# r = requests.post(
#     "https://api.deepai.org/api/sentiment-analysis",
#     data={
#         'text': 'YOUR_TEXT_HERE',
#     },
#     headers={'api-key': 'quickstart-QUdJIGlzIGNvbWluZy4uLi4K'}
# )
# print(r.json())

import requests
import pandas as pd
import threading
from utils.data_evaluator import Evaluator
import time
import sys
# export PYTHONPATH="$PYTHONPATH:$HOME/Ubuntu/sentiment_analysis/" to get data_evaluator until Ramon makes it work

class DeepAi:
    def __init__(self, path_to_data):
        self.URI = "https://api.deepai.org/api/sentiment-analysis"
        self.dataFrame = pd.read_csv(path_to_data)
        self.dataFrame.drop(self.dataFrame.columns[0], axis=1, inplace=True)
        self.nrows = self.dataFrame.shape[0]

    def postToApi(self, text_string):
        print("postingToApi")
        r = requests.post(
            self.URI,
            data={
                'text': text_string,
            },
            headers={'api-key': '32f60ff7-bd13-4c98-8bdc-b5e71b93e067'}
        )
        json = r.json()
        try:
            return [x.lower() for x in json['output']]
        except:
            raise Exception('ERROR: {}, text_string: {}'.format(json, text_string))

    # prepares text for sending through API. This means replacing every dot with ";" as the engine analyzes each sentence separately
    def prepareText(self, text_string):
        return text_string.replace(".", ";")[:-1]

    def getSentimentForLine(self, numLine):
        line = self.dataFrame.iloc[numLine]
        text = line['text']
        # text = self.prepareText(text)
        
        csvSentiment = line['polarity']
        apiSentiments = self.postToApi(text)
        print(apiSentiments)

        apiSentiments = ['positive' if (x == 'verypositive' or x == 'positive') else 'negative' if (x == 'verynegative' or x == 'negative') else 'neutral' for x in apiSentiments]
        print(apiSentiments)

        apiSentiment = 'positive' if (apiSentiments.count('positive') >= apiSentiments.count('negative')) else 'negative'

        print("DeepAi::getSentimentForLine: csvSentiment = {}, apiSentiment = {}".format(csvSentiment, apiSentiment))
        return csvSentiment, apiSentiment

    def getSentimentForMultipleLines(self, numLines):
        lines = self.dataFrame.iloc[numLines]
        texts = lines['text']
        print("len texts 1:", len(texts))

        csvSentiments = lines['polarity']
        texts = [self.prepareText(x) for x in texts]
        print("len texts 2:", len(texts))
        
        globbedText = '. '.join(texts)
        print(globbedText)
        apiSentiments = self.postToApi(globbedText)

        return csvSentiments, apiSentiments


    def worker(self, workerNum, lineNumToTest, expectedList, actualList):
        print("{}: Working on line: {}".format(workerNum, lineNumToTest))
        try:
            # csvSentiment, apiSentiment in outputList
            expectedList[workerNum], actualList[workerNum] = self.getSentimentForLine(lineNumToTest)
        except Exception as error:
            expectedList[workerNum] = 'error'
            actualList[workerNum] = 'error'
            print("{}: ERROR ON LINE: {}. Error message: {}".format(workerNum, lineNumToTest, error))

        return

    def runParallelForLines(self, lineNumbers):
        threads = []
        expectedList = [0 for x in range(len(lineNumbers))]
        actualList = [0 for x in range(len(lineNumbers))]
        
        for i in range(0,len(lineNumbers)):
            t = threading.Thread(target=self.worker, args=(i, lineNumbers[i], expectedList, actualList))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        

        return expectedList, actualList
        
    # deprecated - will be removed in the future
    def getStatisticsForLines(self, list_of_numLines):
        lenAllLines = len(list_of_numLines)
        numOfTrues = 0
        numOfErrors = 0
        numOfNeutrals = 0

        for i, numLine in enumerate(list_of_numLines):
            try:
                csvSentiment, apiSentiment = self.getSentimentForLine(numLine)
            except:
                numOfErrors = numOfErrors + 1
                continue
            
            if (apiSentiment == "verynegative"): apiSentiment = "negative"
            elif (apiSentiment == "verypositive"): apiSentiment = "positive"
            elif (apiSentiment == "neutral"):
                numOfNeutrals = numOfNeutrals + 1
                continue

            same = True if csvSentiment == apiSentiment else False
            if (same):
                numOfTrues = numOfTrues + 1

            print("Working on line: {}/{}. Did match? {}".format(i, lenAllLines, same))

        print("===========\nStatistics:\nnumOfTrues: {}\nnumOfAllLines: {}\nnumOfErors: {}\nnumOfNeutrals: {}\n\npercentMatching: {}"
        .format(numOfTrues, lenAllLines, numOfErrors, numOfNeutrals, numOfTrues/lenAllLines))


    def getBalancedNrows(self, numOfRows):
        temp = list(range(0, numOfRows//2)) + list(range(self.nrows-numOfRows//2, self.nrows))
        print("DUPADUPADUPADUPADUAPDAPUPDA {}".format(temp))
        return temp


def main(numberOfValuesToCheck, numOfThreads):
    pd.set_option('display.max_colwidth', -1)
    deepAi = DeepAi("./test_set/test_set.csv")
    evaluator = Evaluator()
    
    # temporary, need to get values randomly
    # numbersToCheck = list(range(0, numberOfValuesToCheck))
    numbersToCheck = deepAi.getBalancedNrows(numberOfValuesToCheck)
    
    expectedList = []
    actualList = []
    startTime = time.time()
    for i in range(numberOfValuesToCheck // numOfThreads):
        expectedListTemp, actualListTemp = deepAi.runParallelForLines(numbersToCheck[i*numOfThreads:(i+1)*numOfThreads])
        expectedList = expectedList + expectedListTemp
        actualList = actualList + actualListTemp
    
    stopTime = time.time()
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for i in range (len(expectedList)):
        # with open("dupa.txt", 'w+') as f:
        #     f.write("error on line {}, line is: {}".format(numbersToCheck[i], deepAi.dataFrame[0:1]))
        if (expectedList[i] == actualList[i] and expectedList[i] == 'positive'): tp = tp + 1
        elif (expectedList[i] == actualList[i] and expectedList[i] == 'negative'): tn = tn + 1
        elif (expectedList[i] == 'positive' and actualList[i] == 'negative'): fn = fn + 1
        elif (expectedList[i] == 'negative' and actualList[i] == 'positive'): fp = fp + 1 
        elif (expectedList[i] == 'error'):
            try:
                with open("dupa.txt", 'a') as f:
                    f.write("error on line {}, line is:\n{}\n".format(numbersToCheck[i], deepAi.dataFrame[numbersToCheck[i]:numbersToCheck[i]+1]))
            except:
                print("HUGE ERROR")
                continue

    expectedListWithoutErrors = [x for x in expectedList if x != 'error']
    actualListWithoutErrors = [x for x in actualList if x != 'error']

    numOfErrors = len(expectedList) - len(expectedListWithoutErrors)
    timeItTook = stopTime - startTime

    evaluator.evaluate(expectedListWithoutErrors, actualListWithoutErrors, printConfusionMatrix=True)
    print("tp = {}, tn = {}, fp = {}, fn = {}, num of errors = {}, it took {}s".format(tp, tn, fp, fn, numOfErrors, timeItTook))
    return

if __name__ == "__main__":
    if(len(sys.argv) == 1):
        main(164, 80)
    elif(len(sys.argv) == 3):
        print("Running with 2 arguments")
        main(int(sys.argv[1]), int(sys.argv[2]))
    else:
        print("something wrong with passed arguments")
    # deepAi = DeepAi("./test_set/test_set.csv")

    # csvSentiment, apiSentiment = deepAi.getSentimentForLine(4)
    # print("Line 4, Actual sentiment:", csvSentiment, "\nRead sentiment:", apiSentiment)
    # same = True if csvSentiment == apiSentiment else False
    # print("Are the same?", str(same))

    # csvSentiment, apiSentiment = deepAi.getSentimentForLine(279001)
    # print("Line 299001, Actual sentiment:", csvSentiment, "\nRead sentiment:", apiSentiment)
    # same = True if csvSentiment == apiSentiment else False
    # print("Are the same?", str(same))
    
