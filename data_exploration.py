from data_loader import DataLoader
from config import Config
from pprint import pprint
from nltk.tokenize import RegexpTokenizer
import os
import pandas as pd
import code
import xml.etree.ElementTree as ET
import numpy as np
import json

PROP = "CURRENT_DATA"
VALUE = "./data/Amazon_Instant_Video/Amazon_Instant_Video.neg.0.xml"
PATH = "data_path"
COLUMNS = ["id", "domain", "polarity", "summmary", "text"]

"""
Istnieje już podział klas na tematy. Np książki, muzyka etc... Inne słownictwo wobec różnych tematów? Często występujące frazy/schematy oceniania?

Na czym ma się tutaj skupiać analiza eksploracyjna ? Mogę zacząć od utworzenia analizy dla każdej klasy. Być może przenieść dane do csv, ale chyba nie trzeba
Interesujące atrubuty :
    -polaryzacja tekstu
    -długość tekstu
    -długość streszczenia
    -ilość słów w streszczeniu
    -średnia polaryzacja tekstu
    -występowanie znaków specjalnych w tekście? np. &quote
    
Wizualizacja:
    -wykres długości / polaryzacji , czy wypowiedzi nacechowane negatywnie są przeważnie dłuższe czy na odwrót ?

Wnioski:
    -czy istniej korelacja między długością tekstu a jego polaryzacją ? 
    -czy istnieje kolrelacja między długością stresZczenia a polaryzacją ?
    -czy istnieje korelacja między długością streszczenia a długością tekstu ?
    -czy istnieje związek między tematem tekstu a polaryzacją ? lub długością ?
    -czy istnieje związek między występowaniem znaków specjalnych a polaryzacją tekstu lub tematem tekstu ? 


Być może interesująca byłaby agregacja danych nie tylko po temacie, ale np po polaryzacji, albo innym atrybucie jak długość i (w jakiś sposób), badanie korelacji ?
Określenie zmiennych niezależnych i zależnych 

1. To od czego zacząć? Myślę, że od zgrupowania każdego tematu w jeden plik csv, aby uprościć analizę i nie czytać z wielu plików oddziedzlnie.

"""
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class Phrase():
    def __init__(self, id, domain, polarity, summary, text):
        self.id = id
        self.domain = domain
        self.polarity = polarity
        self.summary = summary
        self.text = text

    def toString(self):
        return 'Id: {},\nDomain: {},\nPolarity: {},\nSummary: {},\nText: {} \n'.format(self.id, self.domain, self.polarity, self.summary, self.text)

    def toDictionary(self):
        return {"id": self.id, "domain": self.domain, "polarity": self.polarity, "summary": self.summary, "text": self.text}

class DataExplorer():
    def __init__(self):
        self.config = Config()
        self.domains = []
        self.frames = {}
        self.analyzedDataByDomain = {}
        pass

    def start(self):
        items = os.listdir('./')
        if not 'aggregated' in items:
            os.system('mkdir aggregated')
            return True
        else:
            return False

    def parsePolarityValue(self, value):
        return 1 if value == 'positive' else 0

    def parseData(self):
        loader = DataLoader()
        self.config.addProperty(PROP,VALUE)
        topics = {}
        path = self.config.readValue(PATH)
        domains = os.listdir(path)
        self.domains = domains
        frames = {}

        for topic in domains:
            topics[topic] = []
            for item in os.listdir(path+topic):
                realPath = path + topic + "/" + item
                loader.set_path(realPath)
                try:
                    data = loader.read_xml()
                except ET.ParseError as err:
                    print(err)
                    loader.repair_file(err.position[0], err.position[1])

                for sentance in data:
                    phrase = Phrase(*sentance.toArray())
                    topics[topic].append(phrase.toDictionary())

                frames[topic] = pd.DataFrame(topics[topic])
                frames[topic].to_csv('aggregated/'+topic+'.csv')

            self.frames = frames

    def readData(self):
        frames = {}
        path = self.config.readValue(PATH)
        self.domains = os.listdir(path)
        counter = 0
        for topic in self.domains:
            frames[topic] = pd.read_csv('aggregated/'+topic+'.csv', delimiter=',')
            # counter += 1
            # if(counter == 2):
            #     break

        self.frames = frames

    def getFrames(self):
        return self.frames

    def getDomains(self):
        return self.domains

    def setResultsValue(self, results, value, arr, function):
        try:
            results[value] = function(arr)
        except TypeError as err:
            print(err)

    def analyzeByDomain(self, domain):
        data = self.getFrames()
        topic = data[domain]
        arr = topic.to_numpy()

        results = {}
        tokenizer = RegexpTokenizer(r'\w+')

        if(domain == 'Electronics'):
            print('dupa')

        # results['domain'] = domain

        self.setResultsValue(results,'domain', 'domain', lambda x: x)
        results['numberOfPositives'] = (arr[:,3] == 'positive').sum()
        results['numberOfNegatives'] = len(arr[:,0]) - results['numberOfPositives']
        results['positiveToNegativeRatio'] =  results['numberOfPositives']/results['numberOfNegatives'] if results['numberOfNegatives'] != 0 else results['numberOfPositives']
        self.setResultsValue(results,'meanTextLengthCharacters', arr, lambda arr: np.sum([len(x) for x in arr[:,5]])/len(arr[:,5]))
        # results['meanTextLengthCharacters'] = np.sum([len(x) for x in arr[:,5]])/len(arr[:,5])
        # results['meanTextLengthWords'] = np.sum([len(tokenizer.tokenize(x)) for x in arr[:,5]])/len(arr[:,5])
        self.setResultsValue(results, 'meanTextLengthWords', arr, lambda arr: np.sum([len(tokenizer.tokenize(x)) for x in arr[:,5]])/len(arr[:,5]))
        results['averageTextLengthWhenPolarityPositiveChars'] = np.sum([len(x[2]) for x in arr[:,3:6] if x[0] == 'positive']) / results['numberOfPositives']
        results['averageTextLengthWhenPolarityPositiveWords'] =  np.sum([len(tokenizer.tokenize(x[2])) for x in arr[:,3:6] if x[0] == 'positive']) / results['numberOfPositives']
        results['averageTextLengthWhenPolarityNegativeChars'] =  np.sum([len(x[2]) for x in arr[:,3:6] if x[0] == 'negative']) / results['numberOfNegatives']
        results['averageTextLengthWhenPolarityNegativeWords'] = np.sum([len(tokenizer.tokenize(x[2])) for x in arr[:,3:6] if x[0] == 'negative']) / results['numberOfNegatives']

        print(results)
        self.analyzedDataByDomain[domain] = results

    def dumpResultsToJSON(self):
        path = self.config.readValue('results_path')
        with open(path, "a") as f:
            json.dump(self.analyzedDataByDomain, f, cls=NpEncoder)
        


if __name__ == "__main__":
    de = DataExplorer()
    if(de.start()):
        de.parseData()
    else:
        de.readData()

    for domain in de.getDomains():
        de.analyzeByDomain(domain)

    de.dumpResultsToJSON()
    
