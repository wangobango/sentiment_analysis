import xml.etree.ElementTree as ET
from config import Config
import lxml


class Sentence:

    def __init__(self, sid, domain, polarity, summary, text):
        self.id = sid
        self.domain = domain
        self.polarity = polarity
        self.summary = summary
        self.text = text

    def toString(self):
        return 'Id: {},\nDomain: {},\nPolarity: {},\nSummary: {},\nText: {} \n'.format(self.id, self.domain, self.polarity, self.summary, self.text)

    def toArray(self):
        return [self.id, self.domain, self.polarity, self.summary, self.text]


class DataLoader:

    def __init__(self):
        self.conf = Config()
        self.path = self.conf.readValue('data_path')
        self.data = None

    def set_path(self, path):
        self.path = path

    def read_xml(self):
        if self.path is None:
            return "path is not set!"
        sentences = []
        parser = ET.XMLParser(encoding="UTF-8", )
        tree = ET.parse(self.path, parser=parser)
        root = tree.getroot()

        for child in root:
            sentences.append(Sentence(child.attrib.get("id"), child.find("domain").text, child.find("polarity").text,
                                      child.find("summary").text, child.find("text").text))

        return sentences
