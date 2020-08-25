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
from parameterSetup import ParameterSetup
from outlierMouseFilter import OutlierMouseFilter
from sdFilter import SDFilter

#---------------
# set up parameters

# for data handling
params = ParameterSetup()
oFilter = OutlierMouseFilter()
sdFilter = SDFilter()

metaDataLineNum4eeg = 100
metaDataLineNum4stage = 100
cueWhereEEGDataStarts = "Time"
cueWhereStageDataStarts = "No.,Epoch"

# for signal processing
windowSizeInSec = params.windowSizeInSec   # size of window in time for estimating the state
samplingFreq = params.samplingFreq   # sampling frequency of data

# for training / test data extraction
# if trainWindowNumOrig = 0, all data is used for training.
### trainWindowNumOrig = 1500
### trainWindowNumOrig = 500
trainWindowNumOrig = 0

# for feature extraction
deltaBand = band(1, 4)
thetaBand = band(6, 9)
targetBands = (deltaBand, thetaBand)
lookBackWindowNum = 6

# for drawing spectrum
wholeBand = params.wholeBand
binWidth4freqHisto = params.binWidth4freqHisto    # bin width in the frequency domain for visualizing spectrum as a histogram

voltageRange = (-300,300)
powerRange = (0, 2 * 10 ** 8)
binnedPowerRange = (0, 2 * 10 ** 9)
stage2color = {'W':'b', 'R':'r', 'S':'k', '2':'m', '3':'c', '4':'c', 'M':'c'}

#----------------
# compute parameters

wsizeInTimePoints = samplingFreq * windowSizeInSec   # window size. data is sampled at 128 Hz, so 1280 sample points = 10 sec.

#---------------
# read files
inpath = params.dataDir
outpath = params.pickledDir

outFiles = listdir(outpath)

# dirNames = listdir(inpath)
# for dirName in dirNames:
    # print('dir_stem = ' + dir_stem + ', dir_extension = ' + dir_extension)

if 1:
    dirName = sys.argv[1]
    dir_stem, dir_extension = splitext(dirName)
    if dirName != '.DS_Store' and dir_extension != '.rar':

        files = listdir(inpath + '/' + dirName)
        files2 = files

        for fileFullName in files:
            # print('fileFullName = ' + fileFullName)
            fileID, file_extension = splitext(fileFullName)
            if file_extension == '.txt':

                if oFilter.isOutlier(fileID):
                    print('file ' + fileID + ' is an outlier.')
                else:
                    pklExistsFlag = 0
                    for outFileName in outFiles:
                        # print("fileID = " + fileID + ", outFileName = " + outFileName)
                        if outFileName == fileID + '.pkl':
                            pklExistsFlag = 1
                            break
                    if pklExistsFlag == 0:
                        print('fileFullName = ' + fileFullName)
                        # if fileID.endswith('OPT') and file_extension == '.txt':
                        fileName4eeg = fileFullName
                        fileName4stage = ''
                        for fileFullName2 in files2:
                            # print('fileFullName2 = ' + fileFullName2)
                            fileID2, file_extension2 = splitext(fileFullName2)
                            # print('fileID2 = ' + fileID2)
                            # if fileID2.startswith(fileID) and fileID2 != fileID:
                            if fileID2.startswith(fileID) and fileFullName2 != fileFullName:
                                fileName4stage = fileFullName2
                        if fileName4stage == '':
                            print('file ' + fileName4eeg + ' does not have a corresponding stage file.')
                        else:
                            print('dirName = ' + dirName + ', fileName4eeg = ' + fileName4eeg + ', fileName4stage = ' + fileName4stage)

                            #------------
                            # read label data (wake, REM, non-REM)

                            # stage_fp = open(inpath + dirName + '/' + fileName4stage, 'r')
                            stage_fp = codecs.open(inpath + '/' + dirName + '/' + fileName4stage, 'r', 'shift_jis')
                            for i in range(metaDataLineNum4stage):    # skip lines that describes metadata
                                line = stage_fp.readline()
                                if line.startswith(cueWhereStageDataStarts):
                                    break

                            stagesL = []
                            durationWindNumsL = []
                            for line in stage_fp:
                                line = line.rstrip()
                                ### elems = line.split('\t')
                                elems = line.split(',')
                                # print('elems[0] = ' + elems[0] + ", elems[1] = " + elems[1])
                                # print('elems[0] = ' + elems[0])
                                stageLabel = elems[3]
                                durationWindNum = elems[4]
                                if stageLabel == "NR":
                                    stageLabel = "S"
                                stagesL.append(stageLabel)
                                durationWindNumsL.append(durationWindNum)

                            stageSeq = []
                            stageColorSeq = []

                            for sID in range(len(stagesL)):
                                repeatedStagesl = [stagesL[sID]] * int(durationWindNumsL[sID])
                                repeatedColors = [stage2color[stagesL[sID]]] * int(durationWindNumsL[sID])
                                stageSeq = stageSeq + repeatedStagesl
                                stageColorSeq = stageColorSeq + repeatedColors

                            #---------------
                            # read eeg data

                            # eeg_fp = open(inpath + dirName + '/' + fileName4eeg, 'r')
                            eeg_fp = codecs.open(inpath + '/' + dirName + '/' + fileName4eeg, 'r', 'shift_jis')
                            for i in range(metaDataLineNum4eeg):    # skip 18 lines that describes metadata
                                line = eeg_fp.readline()
                                if line.startswith(cueWhereEEGDataStarts):
                                    break

                            timestampsL = []
                            eegL = []
                            emgL = []
                            for line in eeg_fp:
                                line = line.rstrip()
                                elems = line.split('\t')
                                if len(elems) > 1:
                                    timestampsL.append(elems[0].split(' ')[2].split(':')[2])
                                    eegL.append(float(elems[1]))
                                    emgL.append(float(elems[2]))

                            eeg = np.array(eegL)
                            emg = np.array(emgL)

                            if sdFilter.isOutlier(eeg):
                                print('file' + fileID + ' is an outlier in terms of mean or std')
                            else:
                                print('  eeg.shape = ' + str(eeg.shape) + ', emg.shape = ' + str(emg.shape))
                                timestamps = np.array(timestampsL)
                                ### samplePointNum = eeg.shape[0]

                                #---------------
                                # normalize eeg and emg
                                eeg = (eeg - np.mean(eeg)) / np.std(eeg)
                                emg = (emg - np.mean(emg)) / np.std(emg)

                                if trainWindowNumOrig == 0:
                                    trainWindowNum = math.floor(eeg.shape[0] / wsizeInTimePoints)
                                else:
                                    trainWindowNum = trainWindowNumOrig

                                trainSamplePointNum = trainWindowNum * wsizeInTimePoints

                                #---------------
                                # compute power spectrum and sort it

                                timeSegments = []
                                eegSegmented = []
                                emgSegmented = []
                                powerSpect = np.empty((0, wsizeInTimePoints), float)   # initialize power spectrum

                                #----------------
                                # extract only for train windows
                                startSamplePoint = 0
                                while startSamplePoint + wsizeInTimePoints <= trainSamplePointNum:
                                    endSamplePoint = startSamplePoint + wsizeInTimePoints
                                    timeSegments.append(list(range(startSamplePoint, endSamplePoint)))
                                    eegSegmented.append(eeg[startSamplePoint:endSamplePoint])
                                    emgSegmented.append(emg[startSamplePoint:endSamplePoint])                                    
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

                                saveData = (sortedFreqs, sortedPowerSpect, timeSegments, eegSegmented, emgSegmented, binnedFreqs4visIndices, stageSeq, freqs4wholeBand, binArray4spectrum)

                                file = open(outpath + '/' + 'spectra.' + fileID + '.pkl', 'wb')
                                pickle.dump(saveData, file)



