from data_loader import DataLoader
from config import Config
from pprint import pprint
import os
import pandas as pd
import code


PROP = "CURRENT_DATA"
VALUE = "./data/Amazon_Instant_Video/Amazon_Instant_Video.neg.0.xml"
PATH = "data_path"

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
        return {"id": self.id, "domain": self.domain, "polarity": self.polarity, "summary": self.summary, "test": self.text}

class DataExplorer():
    def __init__(self):
        self.config = Config()
        pass

    def generateCSV(self,arr):
        pass

    def parseData(self):
        loader = DataLoader()
        self.config.addProperty(PROP,VALUE)
        topics = {}
        path = self.config.readValue(PATH)
        domains = os.listdir(path)
        
        """
        Zapisanie każdej domeny do osobnej csv... Sama analiza już na csv'kach
        """

        for topic in domains:
            topics[topic] = pd.DataFrame()
            for item in os.listdir(path+topic):
                realPath = path + topic + "/" + item
                loader.set_path(realPath)
                # code.interact(local=locals())
                print(item)
                print(realPath)
                loader.set_path(realPath)
                loader.read_xml()
                # topics[topic].append([x.toArray() for x in loader.read_xml()])
            # break
        # code.interact(local=locals())

if __name__ == "__main__":

    de = DataExplorer()
    de.parseData()
