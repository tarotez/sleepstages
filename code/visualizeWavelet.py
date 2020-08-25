from __future__ import print_function
import sys
from os import listdir
import pickle
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from parameterSetup import ParameterSetup
from algorithmFactory import AlgorithmFactory

args = sys.argv
if len(args) > 1:
    option = args[1]
else:
    option = ''

# get params shared by programs
params = ParameterSetup()
extractorType = params.extractorType
waveletWidths = params.waveletWidths
time_step = 1 / params.samplingFreq

factory = AlgorithmFactory(extractorType)
extractor = factory.generateExtractor()

#--------
# load data
fileIDs = ['DBL-NO-D0793-HALO-BL-20161005']
if option == '':
    fileID = fileIDs[0]
else:
    fileID = option

# print('fileID = ' + str(fileID))
dataFileHandler = open(params.pickledDir + '/' + params.eegFilePrefix + '.' + fileID + '.pkl', 'rb')
(eeg, emg, stageSeq, timeStamps) = pickle.load(dataFileHandler)
# print('eeg.shape = ' + str(eeg.shape))
# print('len(stageSeq) = ' + str(len(stageSeq)))

# normalize eeg and emg
global_mu = np.mean(eeg)
global_sigma = np.std(eeg)
# print('in featureExtractionClassical(), eeg.shape = ' + str(eeg.shape))

#----------
# extract time windows from EEG, apply FFT, and bin them
# eegSegment = eeg[startSamplePoint:endSamplePoint]
eegSegment = eeg
### emgSegment = emg[startSamplePoint:endSamplePoint]
# timeStampSegment = timeStamps[startSamplePoint:endSamplePoint]
timeStampSegment = timeStamps
# local_mu = np.mean(eegSegment)
# local_sigma = np.std(eegSegment)
eegSegmentStandardized = (eegSegment - global_mu) / global_sigma
local_mu = np.mean(eegSegmentStandardized)
local_sigma = np.std(eegSegmentStandardized)

# features = extractor.getFeatures(self, eegSegment, timeStampSegment, time_step, local_mu, local_sigma)a
features = extractor.getFeatures(eegSegmentStandardized, timeStampSegment, time_step, local_mu, local_sigma)

print('features.shape = ' + str(features.shape) + ', waveletWidths = ' + str(waveletWidths))
plt.imshow(features, extent=[0, features.shape[1], min(params.waveletWidths), max(params.waveletWidths)], cmap='PRGn', aspect='auto', vmax=abs(features).max(), vmin=-abs(features).max())
plt.show()
