from __future__ import print_function
import pickle
from parameterSetup import ParameterSetup

def writePredictionResults(testFileIDandClassifierIDs, params, y_test, y_pred, resultFileDescription):
    outpath = params.pickledDir
    fileFullPathPrefix = outpath + '/' + 'predictions'
    if resultFileDescription != '':
        fileFullPathPrefix = fileFullPathPrfix + '.' + resultFileFullDescription

    for testFileID, classifierID in testFileIDandClassifierIDs:
        fileFullPath = fileFullPathPrefix + '.testFileID.' + testFileID + '.classifierID.' + classifierID + '.csv'
        with open(fileFullPath, 'w') as f:
            # print('len(y_pred) =', len(y_pred), ', len(y_test) =', len(y_test))
            wNum = len(y_test)
            for i in range(wNum):
                dataLine = y_test[i] + ',' + y_pred[i] + '\n'
                f.write(dataLine)
