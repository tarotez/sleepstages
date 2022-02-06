from __future__ import print_function
from freqAnalysisTools import band
from os import listdir
from os.path import isfile, isdir, join, splitext
import sys
import pickle
import math
import numpy as np
from itertools import groupby
import codecs
from parameterSetup import ParameterSetup
# from outlierMouseFilter import OutlierMouseFilter
from sdFilter import SDFilter

class DataReader:

    def __init__(self):
        params = ParameterSetup()
        self.dataDir = params.dataDir
        # for data handling
        self.metaDataLineNumUpperBound4eeg = params.metaDataLineNumUpperBound4eeg
        self.metaDataLineNumUpperBound4stage = params.metaDataLineNumUpperBound4stage
        self.cueWhereEEGDataStarts = params.cueWhereEEGDataStarts
        self.cueWhereStageDataStarts = params.cueWhereStageDataStarts
        self.eegDir = 'Raw'
        self.stageDir = 'Judge'

    def readAll(self, sys):
        #---------------
        # set up parameters
        # get params shared by programs
        params = ParameterSetup()
        # oFilter = OutlierMouseFilter()
        sdFilter = SDFilter()

        # for signal processing
        windowSizeInSec = params.windowSizeInSec   # size of window in time for estimating the state
        samplingFreq = params.samplingFreq   # sampling frequency of data

        # parameters for using history
        preContextSize = params.preContextSize

        # parameters for making a histogram
        wholeBand = params.wholeBand
        binWidth4freqHisto = params.binWidth4freqHisto    # bin width in the frequency domain for visualizing spectrum as a histogram

        # for reading data
        classifierDir = params.classifierDir
        classifierName = params.classifierName
        samplePointNum = samplingFreq * windowSizeInSec   # window size. data is sampled at 128 Hz, so 1280 sample points = 10 sec.
        time_step = 1 / samplingFreq
        binNum4spectrum = round(wholeBand.getBandWidth() / binWidth4freqHisto)
        # print('samplePointNum = ' + str(samplePointNum))
        # past_eeg = np.empty((samplePointNum, 0), dtype = np.float)
        # past_freqHisto = np.empty((binNum4spectrum, 0), dtype = np.float)
        # print('in __init__, past_eeg.shape = ' + str(past_eeg.shape))

        pickledDir = params.pickledDir

        #----------------
        # compute parameters
        wsizeInTimePoints = samplingFreq * windowSizeInSec   # window size. data is sampled at 128 Hz, so 1280 sample points = 10 sec.

        #---------------
        # read files
        outFiles = listdir(pickledDir)
        self.dirName = sys.argv[1]
        print('self.dirName =', self.dirName)
        if len(sys.argv) > 2:
            pickledDir = self.dataDir + '/' + sys.argv[2]

        dir_stem, dir_extension = splitext(self.dirName)
        if dir_extension != '.rar':
            eegPath = self.dataDir + '/' + self.dirName + '/' + self.eegDir
            print('path =', eegPath)
            files = listdir(eegPath) if isdir(eegPath) else []
            print('files =', files)
            stagePath = self.dataDir + '/' + self.dirName + '/' + self.stageDir
            files4stage = listdir(stagePath) if isdir(stagePath) else []
            for fileFullName in files:
                # print('fileFullName = ' + fileFullName)
                fileID, file_extension = splitext(fileFullName)
                if file_extension == '.txt' or file_extension == '.csv':
                    # if oFilter.isOutlier(fileID):
                    #    print('file ' + fileID + ' is an outlier.')
                    # else:
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
                            # if True:
                            for fileFullName4stage in files4stage:
                                # print('fileFullName4stage = ' + fileFullName4stage)
                                fileID4stage, _ = splitext(fileFullName4stage)
                                # print('fileName4eeg = ' + fileName4eeg + ', fileID2 = ' + fileID2)
                                # if fileID2.startswith(fileID) and fileID2 != fileID:
                                if fileID4stage.startswith(fileID):
                                    fileName4stage = fileFullName4stage
                            dirFullName = self.dataDir + '/' + self.dirName
                            eegFilePath = dirFullName + '/' + self.eegDir + '/' + fileName4eeg
                            eeg, emg, timeStamps = self.readEEG(eegFilePath)
                            # if fileName4stage == '' and len(sys.argv) == 2:
                            if fileName4stage == '':
                                print('file ' + fileName4eeg + ' does not have a corresponding stage file, so labels all stages by ?.')
                                stageSeq = ['?' for _ in range(len(timeStamps))]
                            else:
                                print('self.dirName = ' + self.dirName + ', fileName4eeg = ' + fileName4eeg + ', fileName4stage = ' + fileName4stage)
                                #---------------
                                # read eeg and stages
                                stageFilePath = dirFullName + '/' + self.stageDir + '/' + fileName4stage
                                stageSeq = self.readStageSeq(stageFilePath)

                            #---------------
                            # write data
                            if sdFilter.isOutlier(eeg):
                                print('file' + fileID + ' is an outlier in terms of mean or std')
                            else:
                                # eeg = (eeg - np.mean(eeg)) / np.std(eeg)
                                # emg = (emg - np.mean(emg)) / np.std(emg)
                                # if fileName4stage == '':
                                #    saveData = (eeg, emg, timeStamps)
                                #    outpath = open(pickledDir + '/eegOnly.' + fileID + '.pkl', 'wb')
                                #else:
                                #    saveData = (eeg, emg, stageSeq, timeStamps)
                                #    outpath = open(pickledDir + '/eegAndStage.' + fileID + '.pkl', 'wb')
                                saveData = (eeg, emg, stageSeq, timeStamps)
                                outpath = open(pickledDir + '/eegAndStage.' + fileID + '.pkl', 'wb')
                                pickle.dump(saveData, outpath)
                                outpath.close()

    #---------------
    # read stageSeq
    def readStageSeq(self, filePath):
        stage_fp = codecs.open(filePath, 'r', 'shift_jis')
        for i in range(self.metaDataLineNumUpperBound4stage):    # skip lines that describes metadata
            line = stage_fp.readline()
            if line.startswith(self.cueWhereStageDataStarts):
                break
        stagesL = []
        durationWindNumsL = []
        # lineCnt = 0
        for line in stage_fp:
            # if lineCnt < 5:
            #    print('line = ' + line)
            # lineCnt += 1
            line = line.rstrip()
            ### elems = line.split('\t')
            if ',' in line:
                elems = line.split(',')
            else:
                elems = line.split('\t')
            # print('   elems[0] = ' + elems[0] + ", elems[1] = " + elems[1])
            # print('elems[0] = ' + elems[0])
            # print('   elems[3] = ' + elems[3] + ", elems[4] = " + elems[4])
            stageLabel = elems[2]
            durationWindNum = 1
            stageLabel = stageLabel.replace('*','')
            if stageLabel == 'NR':
                stageLabel = 'S'
            if stageLabel == '2':
                stageLabel = 'S'
            stagesL.append(stageLabel)
            durationWindNumsL.append(durationWindNum)

        stageSeq = []
        stageColorSeq = []

        for sID in range(len(stagesL)):
            # print('durationWindNumsL[sID] = ' + str(durationWindNumsL[sID]))
            repeatedStagesl = [stagesL[sID]] * int(durationWindNumsL[sID])
            stageSeq = stageSeq + repeatedStagesl

        # print('stageSeq[0:7] = ' + str(stageSeq[0:7]))
        return stageSeq

    #---------------
    # read eeg and emg data
    def readEEG(self, filePath):
        ### eeg_fp = codecs.open(filePath, 'r', 'shift_jis')
        ### eeg_fp = codecs.open(filePath, 'r', 'shift_jis')
        eeg_fp = codecs.open(filePath, 'r')
        # for i in range(self.metaDataLineNumUpperBound4eeg):    # skip 18 lines that describes metadata
        #    line = eeg_fp.readline()
        #    # print('line = ' + line)
        #    if line.startswith(self.cueWhereEEGDataStarts):
        #        break

        timeStampsL = []
        eegL = []
        emgL = []
        for line in eeg_fp:
            line = line.rstrip()
            # print('line = ' + line)
            if ',' in line:
                elems = line.split(',')
            else:
                elems = line.split('\t')
            if len(elems) > 1:
                ### timeStampsL.append(elems[0].split(' ')[2].split(':')[2])
                # print('  elems[1] = ' + str(elems[1]) + ', elems[2] = ' + str(elems[2]))
                ### timeStampsL.append(elems[0].split(' ')[2])
                timeStampsL.append(elems[0])
                eegL.append(float(elems[1]))
                if len(elems) > 2:
                    emgL.append(float(elems[2]))

        eeg = np.array(eegL)
        emg = np.array(emgL)
        print('  eeg.shape = ' + str(eeg.shape) + ', emg.shape = ' + str(emg.shape))
        timeStamps = np.array(timeStampsL)
        ### samplePointNum = eeg.shape[0]
        return eeg, emg, timeStamps
