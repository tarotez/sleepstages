from __future__ import print_function
from freqAnalysisTools import band
from os import listdir
from os.path import isfile, join, splitext
import sys
import pickle
import math
import numpy as np
from itertools import groupby
import codecs

#---------------
# set up parameters

# for signal processing
wsizeInSec = 10   # size of window in time for estimating the state
samplingFreq = 128   # sampling frequency of data

# for using history
preContextSize = 10
postContextSize = 0

# for training / test data extraction
# if trainWindowNumOrig = 0, all data is used for training.
### trainWindowNumOrig = 1500
### trainWindowNumOrig = 500
trainWindowNumOrig = 0
testSetID = 0

# for feature extraction
deltaBand = band(0.5, 4)
thetaBand = band(6, 9)
targetBands = (deltaBand, thetaBand)
lookBackWindowNum = 6

# for drawing spectrum
wholeBand = band(0, 16)
binWidth4freqHisto = 0.5    # bin width in the frequency domain for visualizing spectrum as a histogram
voltageRange = (-300,300)
powerRange = (0, 2 * 10 ** 8)
binnedPowerRange = (0, 2 * 10 ** 9)

#----------------
# compute parameters

wsizeInTimePoints = samplingFreq * wsizeInSec   # window size. data is sampled at 128 Hz, so 1280 sample points = 10 sec.

#---------------
# read files
classifierDir = '.'
daqDir = '../data/daq'
fileName4eeg = 'elonged_daq_sample.tsv'

paramID = 0
classifierFileName = open(classifierDir + '/' + 'classifier.pkl', 'rb')
classifier =  pickle.load(classifierFileName)

#---------------
# read eeg data

# eeg_fp = open(inpath + dirName + '/' + fileName4eeg, 'r')
eeg_fp = codecs.open(daqDir + '/' + fileName4eeg, 'r', 'shift_jis')
# for i in range(metaDataLineNum4eeg):    # skip 18 lines that describes metadata
#     line = eeg_fp.readline()
#    if line.startswith(cueWhereEEGDataStarts):
#        break

timestampsL = []
eegL = []
for line in eeg_fp:
    line = line.rstrip()
    elems = line.split('\t')
    timestampsL.append(elems[0].split(':')[2])
    eegL.append(elems[1])

eeg = np.array(eegL)
print('  eeg.shape = ' + str(eeg.shape))
timestamps = np.array(timestampsL)
### samplePointNum = eeg.shape[0]

if trainWindowNumOrig == 0:
    trainWindowNum = math.floor(eeg.shape[0] / wsizeInTimePoints)
else:
    trainWindowNum = trainWindowNumOrig

print('eeg.shape[0] = ' + str(eeg.shape[0]) + ', wsizeInTimePoints = ' + str(wsizeInTimePoints))
print('trainWindowNum = math.floor(eeg.shape[0] / wsizeInTimePoints) = ' + str(trainWindowNum))

trainSamplePointNum = trainWindowNum * wsizeInTimePoints

#---------------
# compute power spectrum and sort it

timeSegments = []
eegSegmented = []
powerSpect = np.empty((0, wsizeInTimePoints), float)   # initialize power spectrum

#----------------
# extract only for train windows
startSamplePoint = 0
while startSamplePoint + wsizeInTimePoints <= trainSamplePointNum:
    endSamplePoint = startSamplePoint + wsizeInTimePoints
    timeSegments.append(list(range(startSamplePoint, endSamplePoint)))
    eegSegmented.append(eeg[startSamplePoint:endSamplePoint])
    powerSpect = np.append(powerSpect, [np.abs(np.fft.fft(eeg[startSamplePoint:endSamplePoint])) ** 2], axis = 0)
    startSamplePoint = endSamplePoint

# wNum = powerSpect.shape[0]
time_step = 1 / samplingFreq
freqs = np.fft.fftfreq(powerSpect.shape[1], d = time_step)
idx = np.argsort(freqs)
sortedFreqs = freqs[idx]
sortedPowerSpect = powerSpect[:,idx]
freqs4wholeBand = wholeBand.extractPowerSpectrum(sortedFreqs, sortedFreqs)

#---------------
# bin spectrum

binNum4spectrum = round(wholeBand.getBandWidth() / binWidth4freqHisto)
binArray4spectrum = np.linspace(wholeBand.bottom, wholeBand.top, binNum4spectrum + 1)
binnedFreqs4visIndices = np.digitize(freqs4wholeBand, binArray4spectrum, right=False)

#----------------
# extract total power of target bands

sumPowers = np.empty((trainWindowNum, len(targetBands)))
for wID in range(trainWindowNum):
    for bandID in range(len(targetBands)):
        sumPowers[wID,bandID] = targetBands[bandID].getSumPower(sortedFreqs, sortedPowerSpect[wID,:])

#----------------
# normalize power using total power of all bands

normalizedPowers = np.empty((trainWindowNum, len(targetBands)))
for wID in range(trainWindowNum):
    totalPower = wholeBand.getSumPower(sortedFreqs, sortedPowerSpect[wID,:])
    for bandID in range(len(targetBands)):
        normalizedPowers[wID,bandID] = sumPowers[wID,bandID] / totalPower

#----------------
# sum over past windows

print('trainWindowNum = ' + str(trainWindowNum) + ', lookBackWindowNum = ' + str(lookBackWindowNum))
sumPowersWithPast = np.empty((trainWindowNum - lookBackWindowNum, len(targetBands)))
for wID in range(trainWindowNum - lookBackWindowNum):
    for bandID in range(len(targetBands)):
        sumPowersWithPast[wID,bandID] = sumPowers[(wID - lookBackWindowNum):wID+1,bandID].sum()

#----------------
# extract max power in the target band

maxPowers = np.empty((trainWindowNum, len(targetBands)))
for wID in range(trainWindowNum):
    for bandID in range(len(targetBands)):
        maxPowers[wID,bandID] = targetBands[bandID].getMaxPower(sortedFreqs, sortedPowerSpect[wID,:])

#-----------------
# extract features

featureHistWithoutContext_L = []
featureHistWithContext_L = []
featureThetaDelta_L = []

uniqueBinIDs = np.unique(binnedFreqs4visIndices)
spectralBinNum = uniqueBinIDs.shape[0]
histoMat = np.empty((spectralBinNum, 0), float)
thetaDeltaMat = np.empty((2, 0), float)

# make a feature vector from only one window
wNum = sortedPowerSpect.shape[0]
print('  sortedPowerSpect.shape = ' + str(sortedPowerSpect.shape))

# for wID in range(wNum):
#    powerSpect4show = wholeBand.extractPowerSpectrum(sortedFreqs, sortedPowerSpect[wID,:])
#    histo = np.array([], dtype = np.float)
#    for key, items in groupby (zip(binnedFreqs4visIndices, powerSpect4show), lambda i: i[0]):
#
#        powerSum = np.sum(np.array([x for x in itemsA[:,1]]))
#        histo = np.r_[histo, powerSum]   # catenate powerSum (scalar) to histo (vector)
#    histoMat = np.c_[histoMat, histo]
# featureHistWithoutContext_L.append(histoMat)
# featureThetaDelta_L.append(np.transpose(sumPowers))

# make a feature vector that contains context windows
histoMatWithContext = np.empty((spectralBinNum * (preContextSize + postContextSize + 1), 0), float)
if postContextSize == 0:
    range4wIDcenter = range(wNum)[preContextSize:]
else:
    range4wIDcenter = range(wNum)[preContextSize:-postContextSize]

for wIDcenter in range4wIDcenter:
    histoWithContext = np.array([], dtype = np.float)
    for offset in range(-preContextSize, postContextSize+1):
        powerSpect4show = wholeBand.extractPowerSpectrum(sortedFreqs, sortedPowerSpect[wIDcenter + offset,:])
        for key, items in groupby (zip(binnedFreqs4visIndices, powerSpect4show), lambda i: i[0]):
            itemsA = np.array(list(items))
            powerSum = np.sum(np.array([x for x in itemsA[:,1]]))
            # print('powerSum = ' + str(powerSum))
            histoWithContext = np.r_[histoWithContext, powerSum]

    histoMatWithContext = np.c_[histoMatWithContext, histoWithContext]

featureHistWithContext_L.append(histoMatWithContext)

#---------------
# predict using the trained classifier

featureList = featureHistWithContext_L
feature4test = featureList[testSetID]
print('feature4test = ' + str(feature4test))

x_test = np.transpose(feature4test)
y_pred = classifier.predict(x_test)

print('y_pred = ' + str(y_pred))

