from __future__ import print_function
import pickle
from parameterSetup import ParameterSetup

def writePredictionResults(fileIDpair, params, y_test, y_pred, resultFileDescription):

    testFileID = fileIDpair[0]
    classifierID = fileIDpair[1]
    outpath = params.pickledDir

    fileFullPath = outpath + '/' + 'predictions'
    if resultFileDescription != '':
         fileFullPath = fileFullPath + '.' + resultFileFullDescription
    fileFullPath = fileFullPath + '.testFileID.' + testFileID + '.classifierID.' + classifierID + '.csv'
    file = open(fileFullPath, 'w')
    print('len(y_pred) =', len(y_pred), ', len(y_test) =', len(y_test))
    wNum = len(y_test)
    for i in range(wNum):
        dataLine = y_test[i] + ',' + y_pred[i] + '\n'
        file.write(dataLine)
