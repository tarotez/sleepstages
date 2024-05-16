from __future__ import print_function
import sys
from os import listdir
from os.path import splitext
import numpy as np
from parameterSetup import ParameterSetup
from classifierTrainer import trainClassifier

args = sys.argv
if len(args) > 1:
    option = int(args[1])
else:
    option = -1
fileMax = option

params = ParameterSetup()
files =  listdir(params.batchEvalDir)
fileCnt = 0
for fileFullName in files:
    print('fileFullName = ' + fileFullName)
    fileID, file_extension = splitext(fileFullName)
    print('extension = ' + file_extension)
    if file_extension == '.json':
        paramDir = params.batchEvalDir
        paramFileName = fileFullName
        outputDir = params.batchEvalDir + '/' + fileID
        print(' ')
        print('**** evaluates using ' + paramDir + '/' + paramFileName)
        print(' ')
        trainClassifier(fileMax, paramDir, paramFileName, outputDir)
