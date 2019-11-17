import json
import sys
from .config import Config
from pprint import pprint

class ResultsProcessor:
    def __init__(self):
        self.config = Config()

    def loadResults(self):
        path = self.config.readValue("results_path")
        with (open(path, "r")) as f:
            self.results = json.load(f)
    
    def getValueFromAllDomains(self, value):
        results = {}
        for (key, val) in self.results.items():
            results[key] = val[value]
        pprint(results)
        return results


if __name__ == "__main__":
    rp = ResultsProcessor()
    rp.loadResults()
    rp.getValueFromAllDomains(sys.argv[1])