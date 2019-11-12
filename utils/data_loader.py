import xml.etree.ElementTree as ET
from .config import Config
import lxml


class Sentence:

    def __init__(self, sid, domain, polarity, summary, text):
        self.id = sid
        self.domain = domain
        self.polarity = polarity
        self.summary = summary
        self.text = text

    def toString(self):
        return 'Id: {},\nDomain: {},\nPolarity: {},\nSummary: {},\nText: {} \n'.format(self.id, self.domain,
                                                                                       self.polarity, self.summary,
                                                                                       self.text)

    def toArray(self):
        return [self.id, self.domain, self.polarity, self.summary, self.text]


class DataLoader:

    def __init__(self):
        self.conf = Config()
        self.path = self.conf.readValue('data_path')
        self.data = None

    def set_path(self, path):
        self.path = path

    def repair_file(self, line_nr, column_nr):
        fin = open(self.path, "r").readlines()
        fout = open(self.path, "w")
        i = 0
        for line in fin:
            i += 1
            if i == line_nr:
                new_line = line[0:column_nr - 1] + line[column_nr + 1:]
                fout.write(new_line)
                continue
            fout.write(line)

    def repair_encoding(self):
        fin = open(self.path, "r").readlines()
        fout = open(self.path, "w")
        for line in fin:
            fout.write(self.repair_text(line))

    def repair_text(self, text):
        text = text.replace("&amp;amp;", "&amp;")
        text = text.replace("&amp;eacute;", "&eacute;")
        text = text.replace("&amp;quot;", "&quot;")
        text = text.replace("&amp;lt;", "&lt;")
        text = text.replace("&amp;#34;", "&quot;")
        text = text.replace("&amp;aacute;", "&aacute;;")
        text = text.replace("&amp;gt;", "&gt;")
        return text

    def read_xml(self):
        if self.path is None:
            raise Exception("path is not set!")
        self.repair_encoding()
        sentences = []
        parser = ET.XMLParser(encoding="utf-8")

        try:
            tree = ET.parse(self.path, parser=parser)
        except ET.ParseError as err:
            raise err

        root = tree.getroot()
        for child in root:
            sentences.append(
                Sentence(child.attrib.get("id"), child.find("domain").text.strip(), child.find("polarity").text.strip(),
                         child.find("summary").text.strip(), child.find("text").text.strip()))

        return sentences



# Example of dealing with reading dataset.
#
#
# def main():
#     data_loader = DataLoader()
#     data_loader.set_path(
#         '/home/jacek/Downloads/inzynierka/projekt/datasets/Amazon_Instant_Video/Amazon_Instant_Video.neg.0.xml')
#
#     data_loader.repair_encoding()

    # try:
    #     data_loader.read_xml()
    # except ET.ParseError as err:
    #     print(err)
    #     data_loader.repair_file(err.position[0], err.position[1])


# if __name__ == "__main__":
#     main()
