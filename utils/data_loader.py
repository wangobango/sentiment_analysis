import xml.etree.ElementTree as ET
import html
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


class Loader:
    __parser = None
    __path = None

    def __init__(self):
        self.conf = Config()
        self.__path = self.conf.readValue('data_path')

    def set_parser(self, parser):
        self.__parser = parser

    def set_path(self, path):
        self.__path = path

    def repair_encoding(self):
        fin = open(self.__path, "r").readlines()
        fout = open(self.__path, "w")
        for line in fin:
            fout.write(self.repair_text(line))
        return self

    def repair_text(self, text):
        text = text.replace("&amp;amp;", "&amp;")
        text = text.replace("&amp;eacute;", "&eacute;")
        text = text.replace("&amp;quot;", "&quot;")
        text = text.replace("&amp;lt;", "&lt;")
        text = text.replace("&amp;#34;", "&quot;")
        text = text.replace("&amp;aacute;", "&aacute;;")
        text = text.replace("&amp;gt;", "&gt;")
        return text

    def repair_file(self):
        parser = ET.XMLParser(encoding="utf-8")

        while True:
            try:
                ET.parse(self.__path, parser=parser)
            except ET.ParseError as err:
                print("repairing file at position: " + str(err.position[0]) + ":" + str(err.position[1]))
                self.delete_wrong_char(err.position[0], err.position[1])
            else:
                break

        return self

    def delete_wrong_char(self, line_nr, column_nr):
        fin = open(self.__path, "r").readlines()
        fout = open(self.__path, "w")
        i = 0
        for line in fin:
            i += 1
            if i == line_nr:
                print(line[column_nr])
                new_line = line[0:column_nr - 1] + line[column_nr + 1:]
                fout.write(new_line)
                continue
            fout.write(line)

    def load(self):
        if self.__path is None:
            raise Exception("path is not set!")

        if self.__parser is None:
            raise Exception("parser is not set!")

        return self.__parser.read(self.__path)


class Parser:
    def read(self, path): pass


class PolarityParser(Parser):

    def read(self, path):
        sentences = []
        parser = ET.XMLParser(encoding="utf-8")

        try:
            tree = ET.parse(path, parser=parser)
        except ET.ParseError as err:
            raise err

        root = tree.getroot()
        for child in root:
            sentences.append(
                Sentence(child.attrib.get("id"), child.find("domain").text.strip(), child.find("polarity").text.strip(),
                         html.unescape(child.find("summary").text.strip()), html.unescape(child.find("text").text.strip())))

        return sentences


class AspectParser(Parser):

    def read(self, path):
        return None


# Example of dealing with reading dataset.

# def main():
#     loader = Loader()
#     loader.set_path(
#         '/home/jacek/Downloads/inzynierka/projekt/datasets/Amazon_Instant_Video/Amazon_Instant_Video.pos.4.xml')
#
#     loader.set_parser(PolarityParser())
#     sentences = loader.repair_file().load()
#
#     for s in sentences:
#         print(s.text + "\n")
    # data_loader.repair_encoding()
    #
    # try:
    #     data_loader.read_xml()
    # except ET.ParseError as err:
    #     print(err)
    #     data_loader.repair_file(err.position[0], err.position[1])

# if __name__ == "__main__":
#     main()
