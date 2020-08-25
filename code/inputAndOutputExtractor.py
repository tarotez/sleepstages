from __future__ import print_function
import pickle
import numpy as np
from parameterSetup import ParameterSetup

class InputAndOutputExtractor(object):

    def __init__(self):
        self.params = ParameterSetup()
        self.eegDir = self.params.eegDir
        self.featureDir = self.params.featureDir
        self.classifierDir = self.params.classifierDir
        self.extractorType = self.params.extractorType

    def extract(self, fileID):
        dataPath = open(self.eegDir + '/eegAndStage.' + fileID + '.pkl', 'rb')
        (eeg, emg, stageSeq, timeStamps) = pickle.load(dataPath)
        if self.params.useEMG:
            emgLabel = 'withEMG'
        else:
            emgLabel = 'withoutEMG'
        featurePath = self.featureDir + '/features.' + self.extractorType + '.' + emgLabel + '.' + fileID + '.pkl'
        # print('featurePath = ' + featurePath)
        featureFileHandler = open(featurePath, 'rb')
        features = pickle.load(featureFileHandler)
        fLen = features.shape[0]
        sLen = len(stageSeq)
        # print('sampleNum = ' + str(fLen) + ', sLen = ' + str(sLen))

        # Below is for the case that not all of the time windows have been labeled.
        # In such a case, stageSeq is shorter than featureArray
        if fLen != sLen:
            features4train = features[:sLen]
            # fLenShort = features4train.shape[1]
            # print('for testSetID = ' + str(testSetID) + ', trainSetID = ' + str(trainSetID) + ' is used.')
            # print(' original length of feature = ' + str(fLen))
            # print(' revised length of feature = ' + str(fLenShort))
            # print(' len(stageSeq_L[trainSetID]) = ' + str(sLen))
            # print('')
        else:
            features4train = features

        x = features4train
        y = np.array(stageSeq)

        inputAndOutput = (x, y)
        return inputAndOutput
