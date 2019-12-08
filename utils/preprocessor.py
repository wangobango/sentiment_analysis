from .data_exploration import DataExplorer
from .data_loader import Loader, PolarityParser
from .config import Config
from console_progressbar import ProgressBar
from data_exploration import Phrase
import os
import pandas as pd


PATH = "data_path"

class Preprocessor:
  def __init__(self):
    self.explorer = DataExplorer()

  def aggregateData(self):
      loader = Loader()
      loader.set_parser(PolarityParser())
      config = Config()
      topics = {}
      path = config.readValue(PATH)
      domains = os.listdir(path)
      self.domains = domains
      pb = ProgressBar(total=int(len(self.domains)-1),prefix='Data parsing in progress', suffix='', decimals=3, length=50, fill='X', zfill='-')
      frames = {}
      data = []

      for idx, topic in enumerate(domains):
          topics[topic] = []
          for item in os.listdir(path+topic):
              realPath = path + topic + "/" + item
              print(realPath)
              loader.set_path(realPath)
              # try:
              data = loader.repair_file().load()
              # except ET.ParseError as err:
              #     if '-debug' in sys.argv:
              #         print(err)
              #     loader.repair_encoding()

              if (len(data) > 0):
                  for sentance in data:
                      phrase = Phrase(*sentance.toArray())
                      topics[topic].append(phrase.toDictionary())
              else:
                  raise Exception('data length is 0')

          frames[topic] = pd.DataFrame(topics[topic])
          frames[topic].to_csv('aggregated/'+topic+'.csv')
          pb.print_progress_bar(idx)
              # print("Done topic: {}, {} / {}".format(topic, idx, len(domains)))
          self.frames = frames
          