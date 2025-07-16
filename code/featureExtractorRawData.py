from __future__ import print_function
from featureExtractor import FeatureExtractor

class FeatureExtractorRawData(FeatureExtractor):

    def __init__(self, params):
        self.params = params
        self.extractorType = 'rawData'

    def getFeatures(self, eegSegment, timeStampSegment, time_step, local_mu, local_sigma):
        return eegSegment
