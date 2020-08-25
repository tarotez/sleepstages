from __future__ import print_function
import numpy as np
from parameterSetup import ParameterSetup
from featureExtractor import FeatureExtractor

class FeatureExtractorRealFourier(FeatureExtractor):

    def __init__(self):
        self.extractorType = 'realFourier'
        params = ParameterSetup()

    def getFeatures(self, eegSegment, timeStampSegment=0, time_step=0, local_mu=0, local_sigma=0):
        fourierTransformed = np.fft.fft(eegSegment)
        return np.array([fourierTransformed])
