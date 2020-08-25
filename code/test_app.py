import json
from config import Config

project_home = Config.project_home

print('project_home = ' + project_home)

pickledDir = project_home + '/pickled'

parameterFilePath = open(pickledDir + '/params.json')
d = json.load(parameterFilePath)
windowSizeInSec = d['windowSizeInSec']

print('wSS = ' + str(windowSizeInSec))
