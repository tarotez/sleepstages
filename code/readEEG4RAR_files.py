from __future__ import print_function
from freqAnalysisTools import band
from os import listdir
from os.path import isfile, join, splitext
import pickle
import math
import numpy as np
from itertools import groupby

#---------------
# set up parameters

# for data handling
metaDataLineNum4eeg = 18
metaDataLineNum4stage = 29

# for signal processing
wsizeInSec = 10   # size of window in time for estimating the state
samplingFreq = 128   # sampling frequency of data

# for training / test data extraction
# if trainWindowNumOrig = 0, all data is used for training.
### trainWindowNumOrig = 1500
### trainWindowNumOrig = 500
trainWindowNumOrig = 0

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
stage2color = {'W':'b', 'R':'r', 'S':'k'}

#----------------
# compute parameters

wsizeInTimePoints = samplingFreq * wsizeInSec   # window size. data is sampled at 128 Hz, so 1280 sample points = 10 sec.

#---------------
# read files
inpath = '../data/RAR files/'
outpath = '../data/pickled/'

files = listdir(inpath)
files2 = files

for fileFullName in files:
    fileID, file_extension = splitext(fileFullName)
    if fileID.endswith('OPT') and file_extension == '.txt':
        fileName4eeg = fileFullName
        fileName4stage = ''
        for fileFullName2 in files2:
            fileID2, file_extension2 = splitext(fileFullName2)
            if fileID2.startswith(fileID) and fileID2 != fileID:
                fileName4stage = fileFullName2
        # print('fileName4eeg = ' + fileName4eeg + ', fileName4stage = ' + fileName4stage)

        #------------
        # read label data (wake, REM, non-REM)

        stage_fp = open(inpath + fileName4stage, 'r')
        for i in range(metaDataLineNum4stage):    # skip 29 lines that describes metadata
            stage_fp.readline()

        stagesL = []
        durationWindNumsL = []
        for line in stage_fp:
            line = line.rstrip()
            elems = line.split('\t')
            stagesL.append(elems[3])
            durationWindNumsL.append(elems[4])

        stageSeq = []
        stageColorSeq = []

        for sID in range(len(stagesL)):
            repeatedStagesl = [stagesL[sID]] * int(durationWindNumsL[sID])
            repeatedColors = [stage2color[stagesL[sID]]] * int(durationWindNumsL[sID])
            stageSeq = stageSeq + repeatedStagesl
            stageColorSeq = stageColorSeq + repeatedColors

        #---------------
        # read eeg data

        eeg_fp = open(inpath + fileName4eeg, 'r')
        for i in range(metaDataLineNum4eeg):    # skip 18 lines that describes metadata
            eeg_fp.readline()

        timestampsL = []
        eegL = []
        for line in eeg_fp:
            line = line.rstrip()
            elems = line.split('\t')
            timestampsL.append(elems[0].split(' ')[2].split(':')[2])
            eegL.append(elems[1])

        eeg = np.array(eegL)
        timestamps = np.array(timestampsL)
        ### samplePointNum = eeg.shape[0]

        if trainWindowNumOrig == 0:
            trainWindowNum = math.floor(eeg.shape[0] / wsizeInTimePoints)
        else:
            trainWindowNum = trainWindowNumOrig

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

        stageColorSeq4train = stageColorSeq[0:trainWindowNum]

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

        saveData = (sumPowers, normalizedPowers, sumPowersWithPast, maxPoｗers, stageColorSeq4train, sortedFreqs, sortedPowerSpect, timeSegments, eegSegmented, binnedFreqs4visIndices, stageSeq, freqs4wholeBand, binArray4spectrum)

        file = open(outpath + fileID + '.pkl', 'wb')
        pickle.dump(saveData, file)



