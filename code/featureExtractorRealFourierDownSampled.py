from __future__ import print_function
import numpy as np
from parameterSetup import ParameterSetup
from featureExtractor import FeatureExtractor
from featureDownSampler import FeatureDownSampler

class FeatureExtractorRealFourierDownSampled(FeatureExtractor):

    def __init__(self):
        self.extractorType = 'realFourier-downsampled'
        params = ParameterSetup()
        self.outputDim = params.downsample_outputDim

    def getFeatures(self, eegSegment, timeStampSegment=0, time_step=0, local_mu=0, local_sigma=0):
        fourierTransformed = np.fft.fft(eegSegment)
        inputTensor = np.array([[fourierTransformed]])
        featureDownSampler = FeatureDownSampler()
        fourierDownsampled = featureDownSampler.downSample(inputTensor, self.outputDim)[0]
        return fourierDownsampled
