from utils.config import Config
import sys,csv

class InputExample():
    def __init__(self, guid, text_a, text_b = None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class BertDataConverter():
    def __init__(self):
        self.config = Config()

    def get_dev_examples(self):
        dev_set = self.read_tsv(self.config.readValue('bert_dev_set'), "dev")
        print(dev_set)

    @staticmethod
    def read_tsv(input_file, set_type,quotechar=None):
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for i,line in enumerate(reader):
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                guid = "%s-%s" % (set_type, i)
                text_a = line[3]
                label = line[1]
                lines.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            return lines

if __name__ == "__main__":
    BertDataConverter().get_dev_examples()