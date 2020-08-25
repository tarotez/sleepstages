from __future__ import print_function
import numpy as np
from scipy import signal
from parameterSetup import ParameterSetup
from featureExtractor import FeatureExtractor
from featureDownSampler import FeatureDownSampler
from algorithmFactory import AlgorithmFactory

class FeatureExtractorMerged(FeatureExtractor):

    def __init__(self, extractorType):
        self.extractorType = extractorType
        params = ParameterSetup()
        self.outputDim = params.downsample_outputDim

    def getFeatures(self, eegSegment, timeStampSegment=0, time_step=0, local_mu=0, local_sigma=0):
        extractorTypes = self.extractorType.split(',')
        extractorCnt = 0
        for extractorType in extractorTypes:
            factory = AlgorithmFactory(extractorType.rstrip().lstrip())
            extractor = factory.generateExtractor()
            # print('eegSegment.shape = ' + str(eegSegment.shape))
            features = extractor.getFeatures(eegSegment)
            # print('****** features.shape = ' + str(features.shape))
            if extractorCnt == 0:
                merged = features
            else:
                merged = np.concatenate((merged, features), axis=0)
            extractorCnt += 1
            # print('****** merged.shape = ' + str(merged.shape))
        # print('########### final merged.shape = ' + str(merged.shape))
        mergedTransposed = merged.transpose()
        return mergedTransposed
