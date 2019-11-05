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

class DeepAi:
    def __init__(self, path_to_data):
        self.URI = "https://api.deepai.org/api/sentiment-analysis"
        self.dataFrame = pd.read_csv(path_to_data)
        self.dataFrame.drop(self.dataFrame.columns[0], axis=1, inplace=True)
        self.nrows = self.dataFrame.shape[0]

    def readLineFromDataFrame(self, numLine):
        return self.dataFrame.iloc[numLine]

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
        print(json)
        return [x.lower() for x in json['output']]

    # prepares text for sending through API. This means replacing every dot with ";" as the engine analyzes each sentence separately
    def prepareText(self, text_string):
        return text_string.replace(".", ";")[:-1]

    def getSentimentForLine(self, numLine):
        line = self.readLineFromDataFrame(numLine)
        text = line['text']
        text = self.prepareText(text)
        
        csvSentiment = line['polarity']
        apiSentiment = self.postToApi(text)
        
        print(csvSentiment, apiSentiment)
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


    def worker(self, i, outputList):
        print("Working on line: {}".format(i))
        try:
            csvSentiment, apiSentiment = self.getSentimentForLine(i)
        except:
            outputList[i] = 'error'
            return
        
        if (apiSentiment == "verynegative"): apiSentiment = "negative"
        elif (apiSentiment == "verypositive"): apiSentiment = "positive"
        elif (apiSentiment == "neutral"):
            outputList[i] = 'neutral'
            return

        same = True if csvSentiment == apiSentiment else False
        if (same):
            outputList[i] = 'true'
        else:
            outputList[i] = 'false'

        print("Did line {} match? {}".format(i, same))
        return

    def runParallel(self):
        threads = []
        outputList = [0 for x in range(200)]
        for i in range(0,200):
            t = threading.Thread(target=self.worker, args=(i, outputList))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        same = outputList.count('true')
        different = outputList.count('false')
        errors = outputList.count('error')
        neutrals = outputList.count('neutral')
        print("Num of same: {}\nNum of diff: {}\nNum of errors: {}\nNum of neutrals: {}".format(same, different, errors, neutrals))
        return
        
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

def main():
    deepAi = DeepAi("./test_set/test_set.csv")
    # deepAi.runParallel()
    # return
    
    for i in range(0,29):
        line = deepAi.dataFrame.iloc[i]
        text = line['text']
        print("i = {}, text = {}".format(i, text))

        text = deepAi.prepareText(text)
        print("i = {}, prepared text = {}\n".format(i, text))
    return
    # numbersToCheck = list(range(0, 10)) + list(range(299000, 299010))
    numbersToCheck = list(range(0,29))
    deepAi.getSentimentForMultipleLines(numbersToCheck)
    # deepAi.getStatisticsForLines(numbersToCheck)


if __name__ == "__main__":
    main()
    # deepAi = DeepAi("./test_set/test_set.csv")

    # csvSentiment, apiSentiment = deepAi.getSentimentForLine(4)
    # print("Line 4, Actual sentiment:", csvSentiment, "\nRead sentiment:", apiSentiment)
    # same = True if csvSentiment == apiSentiment else False
    # print("Are the same?", str(same))

    # csvSentiment, apiSentiment = deepAi.getSentimentForLine(279001)
    # print("Line 299001, Actual sentiment:", csvSentiment, "\nRead sentiment:", apiSentiment)
    # same = True if csvSentiment == apiSentiment else False
    # print("Are the same?", str(same))
    
