from data_loader import DataLoader
from config import Config
from pprint import pprint

PROP = "CURRENT_DATA"
VALUE = "./data/Amazon_Instant_Video/Amazon_Instant_Video.neg.0.xml"

if __name__ == "__main__":
    loader = DataLoader()
    # loader.read_xml()

    conf = Config()
    conf.addProperty(PROP,VALUE)

    loader.set_path(conf.readValue(PROP))

    data = loader.read_xml()
    for item in data:
        pprint(item.toString())