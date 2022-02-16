from __future__ import print_function
from freqAnalysisTools import band
from os import listdir
from os.path import isfile, join, splitext
import sys
import pickle
import math
import numpy as np
from itertools import groupby
from datetime import datetime, timedelta
import codecs
from parameterSetup import ParameterSetup
from outlierMouseFilter import OutlierMouseFilter
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
        self.params_pickledDir = params.pickledDir
        self.samplingFreq_from_params = params.samplingFreq
        self.eegDir = 'Raw'
        self.stageDir = 'Judge'

    def readAll(self, sys):
        oFilter = OutlierMouseFilter()
        sdFilter = SDFilter()

        #---------------
        # read files
        outFiles = listdir(self.params_pickledDir)
        self.dirName = sys.argv[1]
        if len(sys.argv) > 2:
            pickledDir = self.dataDir + '/' + sys.argv[2]
        else:
            pickledDir = self.params_pickledDir

        dir_stem, dir_extension = splitext(self.dirName)
        if dir_extension != '.rar':

            files = listdir(self.dataDir + '/' + self.dirName + '/' + self.eegDir)
            files2 = listdir(self.dataDir + '/' + self.dirName + '/' + self.stageDir)

            for fileFullName in files:
                # print('fileFullName = ' + fileFullName)
                fileID, file_extension = splitext(fileFullName)
                if file_extension == '.txt':
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
                            for fileFullName2 in files2:
                                # print('fileFullName2 = ' + fileFullName2)
                                fileID2, file_extension2 = splitext(fileFullName2)
                                # print('fileName4eeg = ' + fileName4eeg + ', fileID2 = ' + fileID2)
                                # if fileID2.startswith(fileID) and fileID2 != fileID:
                                if fileID2.startswith(fileID):
                                    fileName4stage = fileFullName2
                            if fileName4stage == '' and len(sys.argv) == 2:
                                print('file ' + fileName4eeg + ' does not have a corresponding stage file.')
                            else:
                                print('self.dirName = ' + self.dirName + ', fileName4eeg = ' + fileName4eeg + ', fileName4stage = ' + fileName4stage)

                                #---------------
                                # read eeg and stages
                                dirFullName = self.dataDir + '/' + self.dirName
                                eegFilePath = dirFullName + '/' + self.eegDir + '/' + fileName4eeg
                                eeg, emg, timeStamps = self.readEEG(eegFilePath)
                                if not fileName4stage == '':
                                    stageFilePath = dirFullName + '/' + self.stageDir + '/' + fileName4stage
                                    stageSeq = self.readStageSeq(stageFilePath)

                                #---------------
                                # write data
                                if sdFilter.isOutlier(eeg):
                                    print('file' + fileID + ' is an outlier in terms of mean or std')
                                else:
                                    # eeg = (eeg - np.mean(eeg)) / np.std(eeg)
                                    # emg = (emg - np.mean(emg)) / np.std(emg)
                                    if fileName4stage == '':
                                        saveData = (eeg, emg, timeStamps)
                                        outpath = open(pickledDir + '/eegOnly.' + fileID + '.pkl', 'wb')
                                    else:
                                        saveData = (eeg, emg, stageSeq, timeStamps)
                                        outpath = open(pickledDir + '/eegAndStage.' + fileID + '.pkl', 'wb')
                                    pickle.dump(saveData, outpath)

    #---------------
    # read stageSeq
    def readStageSeq(self, filePath):
        stage_fp = codecs.open(filePath, 'r', 'shift_jis')
        for i in range(self.metaDataLineNumUpperBound4stage):    # skip lines that describes metadata
            line = stage_fp.readline()
            if line.startswith(self.cueWhereStageDataStarts):
                break
            if i == self.metaDataLineNumUpperBound4stage - 1:
                # print('stage file without metadata header, but it\'s okay.')
                stage_fp.close()
                stage_fp = codecs.open(filePath, 'r', 'shift_jis')

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
            # print('line =', line)
            # print('elems =', elems)
            if len(elems) > 1:
                stageLabel = elems[2]
            else:
                stageLabel = elems[0]
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
        eeg_fp = codecs.open(filePath, 'r', 'shift_jis')
        for i in range(self.metaDataLineNumUpperBound4eeg):    # skip 18 lines that describes metadata
            line = eeg_fp.readline()
            # print('line = ' + line)
            if line.startswith(self.cueWhereEEGDataStarts):
                break
            if i == self.metaDataLineNumUpperBound4eeg - 1:
                # print('eeg file without metadata header, but it\'s okay.')
                eeg_fp.close()
                eeg_fp = codecs.open(filePath, 'r', 'shift_jis')
                ### print('metadata (header) for the EEG file is not correct.')
                ## quit()
        #-----------
        # read text file
        print('---------------------')
        print('Started to read ' + filePath + '. It may take a few minutes before starting to classify. Please wait.')
        print('---------------------')
        timeStampsL = []
        eegL = []
        emgL = []
        timeStamp = datetime.now()
        for line in eeg_fp:
            line = line.rstrip()
            # print('line = ' + line)
            if ',' in line:
                elems = line.split(',')
            elif '\t' in line:
                elems = line.split('\t')
            else:
                elems = line.split(' ')
            if len(elems) > 1:
                ### timeStampsL.append(elems[0].split(' ')[2].split(':')[2])
                # print('  elems[1] = ' + str(elems[1]) + ', elems[2] = ' + str(elems[2]))
                if ' ' in elems[0]:
                    timeStampsL.append(elems[0].split(' ')[2])
                else:
                    timeStampsL.append(elems[0])
                eegL.append(float(elems[1]))
                if len(elems) > 2:
                    try:
                        emgL.append(float(elems[2]))
                    except ValueError:
                        emgL.append(0)
            elif len(elems) == 1:
                # when wave data contains no timestamp
                timeStampsL.append(str(timeStamp).split(' ')[-1])
                timeStamp += timedelta(seconds=1.0 / self.samplingFreq_from_params)
                eegL.append(float(elems[0]))
                # print(timeStampsL[-1], ':', eegL[-1])

        eeg = np.array(eegL)
        emg = np.array(emgL)
        timeStamps = np.array(timeStampsL)
        # print('eeg.shape = ' + str(eeg.shape) + ', emg.shape = ' + str(emg.shape))
        # print('eeg[:10] =', eeg[:10])
        # print('timeStamps[:10] =', timeStamps[:10])
        ### samplePointNum = eeg.shape[0]
        return eeg, emg, timeStamps
