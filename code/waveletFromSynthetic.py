from __future__ import print_function
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from parameterSetup import ParameterSetup
from featureExtractorWavelet import FeatureExtractorWavelet

params = ParameterSetup()
waveletWidths = params.waveletWidths
samplingFreq = params.samplingFreq
### samplingFreq = 1000
extractor = FeatureExtractorWavelet()

timeLengthInSec = 10
physicalFreqs = [32]
samplePointNum = timeLengthInSec * samplingFreq
t = np.linspace(0, timeLengthInSec, samplePointNum, endpoint=False)
# syntheticWave  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
syntheticWave = np.cos(2 * np.pi * physicalFreqs[0] * t)

cwtMat = extractor.getFeatures(syntheticWave)
print('cwtMat.shape = ' + str(cwtMat.shape) + ', waveletWidths = ' + str(waveletWidths))
plt.imshow(cwtMat, extent=[0, cwtMat.shape[1], min(params.waveletWidths), max(params.waveletWidths)], cmap='PRGn', aspect='auto', vmax=abs(cwtMat).max(), vmin=-abs(cwtMat).max())
plt.show()
