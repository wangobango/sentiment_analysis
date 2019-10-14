import xml.etree.ElementTree as ET
from config import Config


class Sentence:

    def __init__(self, sid, domain, polarity, summary, text):
        self.id = sid
        self.domain = domain
        self.polarity = polarity
        self.summary = summary
        self.text = text


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
        tree = ET.parse(self.path)
        root = tree.getroot()

        for child in root:
            sentences.append(Sentence(child.attrib.get("id"), child.find("domain").text, child.find("polarity").text,
                                      child.find("summary").text, child.find("text").text))

        return sentences
