import json

class Config:
  
  def __init__(self, path = './config.json'):
    self.path = path
    self.config = self.loadConfig()

  def loadConfig(self):
    data = ''
    with open(self.path, "r") as f:
      for item in f:
        data += item

    return json.loads(data)

  def saveConfig(self):
    with open(self.path, "w") as f:
      json.dump(self.config, f)

  def showConfig(self):
    for (key,value) in self.config.items():
      print('key - {}, value: {}'.format(key,value))

  def showValue(self, prop):
    print(self.config[prop])

  def readValue(self, prop):
    try:
      return self.config[prop]    
    except:
      print('No such prop found!')

  def showDataPath(self):
    self.showValue('data_path')

  def writeValue(self, prop, value):
    self.config[prop] = value
    self.saveConfig()

  def writeDataPath(self, value):
    self.writeValue('data_path', value)

  def addProperty(self, prop, value):
    self.config[prop] = value
    self.saveConfig()

