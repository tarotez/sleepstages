from __future__ import print_function
import numpy as np
from scipy import signal
from parameterSetup import ParameterSetup
from featureExtractor import FeatureExtractor
from featureDownSampler import FeatureDownSampler

class FeatureExtractorWaveletDownSampled(FeatureExtractor):

    def __init__(self):
        self.extractorType = 'wavelet-downsampled'
        params = ParameterSetup()
        self.outputDim = params.downsample_outputDim

    def getFeatures(self, eegSegment, timeStampSegment=0, time_step=0, local_mu=0, local_sigma=0):
        params = ParameterSetup()
        widths = params.waveletWidths
        waveletTransformed = signal.cwt(eegSegment, signal.ricker, widths)
        inputTensor = np.array([waveletTransformed])
        # print('inputTensor.shape = ' + str(inputTensor.shape))
        featureDownSampler = FeatureDownSampler()
        # print('self.outputDim = ' + str(self.outputDim))
        waveletTransformedDownsampled = featureDownSampler.downSample(inputTensor, self.outputDim)[0]
        # print('waveletTransformedDownsampled.shape = ' + str(waveletTransformedDownsampled.shape))
        return waveletTransformedDownsampled
