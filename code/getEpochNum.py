import sys
from os.path import splitext
import pickle
import numpy as np
from parameterSetup import ParameterSetup
from writePredictionResults import writePredictionResults
from fileManagement import getAllEEGFiles, fileIDsFromEEGFiles

params = ParameterSetup()
eegFiles = getAllEEGFiles(params)
fileIDs = fileIDsFromEEGFiles(eegFiles)
epochNumL = []
for fileID in fileIDs:
    print('fileID =', fileID)
    predictionTargetDataFilePath = params.pickledDir + '/' + params.eegFilePrefix + '.' + fileID + '.pkl'
    # print('   predictionTargetDataFilePath =', predictionTargetDataFilePath)
    dataFileHandler = open(predictionTargetDataFilePath, 'rb')
    (eeg, ch2, stageSeq, timeStamps) = pickle.load(dataFileHandler)
    epochNum = len(stageSeq)
    print('epochNum =', epochNum)
    epochNumL.append(epochNum)

print(' ')
print('epochNumL mean =', np.array(epochNumL).mean())
