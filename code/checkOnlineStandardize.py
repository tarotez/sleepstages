from __future__ import print_function
import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from parameterSetup import ParameterSetup
from statistics import recompMean, recompVariance

args = sys.argv
if len(args) > 1:
    option = args[1]
else:
    option = ''

# get params shared by programs
timeLengthInSec = 120
timeWindowSize = 10
params = ParameterSetup()
samplePointNum = params.samplingFreq * timeLengthInSec
wsizeInSamplePointNum = params.samplingFreq * timeWindowSize
timeWindowStrideInSamplePointNum = wsizeInSamplePointNum

# driftCoeff = 0.5
noizeStdDev = 8
# mu = np.linspace(0, driftCoeff * timeLengthInSec, samplePointNum)
t = np.linspace(0, timeLengthInSec, samplePointNum)
driftFreq = 0.05
mu  = np.cos(2 * np.pi * driftFreq * t)
eeg = (np.random.random((samplePointNum)) * noizeStdDev) + mu

eeg_mean = 0
eeg_variance = 0
oldSampleNum = 0
for startSamplePoint in range(0, samplePointNum, timeWindowStrideInSamplePointNum):
    endSamplePoint = startSamplePoint + wsizeInSamplePointNum
    if endSamplePoint > samplePointNum:
        break
    eegSegment = eeg[startSamplePoint:endSamplePoint]
    eeg_old_mean = eeg_mean
    eeg_mean = recompMean(eegSegment, eeg_mean, oldSampleNum)
    eeg_variance = recompVariance(eegSegment, eeg_variance, eeg_old_mean, eeg_mean, oldSampleNum)
    standardized_eegSegment = (eegSegment - eeg_mean) / np.sqrt(eeg_variance)
    if startSamplePoint == 0:
        standardized_eeg = standardized_eegSegment
    else:
        standardized_eeg = np.r_[standardized_eeg, standardized_eegSegment]
    oldSampleNum += eegSegment.shape[0]
    print('oldSampleNum = ' + str(oldSampleNum))

plt.plot(eeg)
plt.plot(standardized_eeg)
plt.show()
