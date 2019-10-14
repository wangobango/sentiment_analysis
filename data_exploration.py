from data_loader import DataLoader
from config import Config
from pprint import pprint
import os
import pandas as pd


PROP = "CURRENT_DATA"
VALUE = "./data/Amazon_Instant_Video/Amazon_Instant_Video.neg.0.xml"

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

def generateCSV(arr):
    pass

if __name__ == "__main__":
    loader = DataLoader()
    # loader.read_xml()

    conf = Config()
    conf.addProperty(PROP,VALUE)

    loader.set_path(conf.readValue(PROP))

    # data = loader.read_xml()
    # for item in data:
    #     pprint(item.toString())

    domains = os.listdir('./data/')
    for topic in domains:
        for item in topic:
            pass
    # print(conf.readValue('data_path') + domains[0])