from __future__ import print_function
import numpy as np
from scipy import signal
from featureExtractor import FeatureExtractor

class FeatureExtractorWavelet(FeatureExtractor):

    def __init__(self, params):
        self.params = params
        self.extractorType = 'wavelet'

    def getFeatures(self, eegSegment, timeStampSegment=0, time_step=0, local_mu=0, local_sigma=0):
        params = self.params
        widths = params.waveletWidths
        waveletTransformed = signal.cwt(eegSegment, signal.ricker, widths)
        return waveletTransformed
