import pickle
import numpy as np
from ksstatistics import StatisticalTester
from fileManagement import readStandardMice, getFileIDs
from parameterSetup import ParameterSetup

eegFilePrefix = 'eegAndStage'
params = ParameterSetup()
pickledDir = params.pickledDir
standardMiceData, standardMiceFileFullNames = readStandardMice()
statisticalTester = StatisticalTester(standardMiceData)
samplingFreq = 128
timeWindowSizeInSec = 10
sampleNumInTimeWindow = samplingFreq * timeWindowSizeInSec

maxSegmentNum = 60

fileIDs = getFileIDs(pickledDir, eegFilePrefix)

dMat_L = []
segmentNums_L = []
fileNum = 0
for fileID in fileIDs:
### for fileID in fileIDs[0:2]:
    fileFullName = eegFilePrefix + '.' + fileID + '.pkl'
    useThisMouse = True
    for standardMiceFileFullName in standardMiceFileFullNames:
        if fileFullName.startswith(standardMiceFileFullName):
            useThisMouse = False
    if useThisMouse:
        dataFileHandler = open(pickledDir + '/' + fileFullName, 'rb')
        (eeg, emg, stageSeq, timeStamps) = pickle.load(dataFileHandler)
        allEEG = eeg
        print('allEEG.shap = ' + str(allEEG.shape))
        dVec_L = []
        ### segmentNum = np.int(np.floor(allEEG.shape[0]/sampleNumInTimeWindow))
        segmentNum = maxSegmentNum
        print('segmentNum = ' + str(segmentNum))
        segmentNums_L.append(segmentNum)
        for segmentID in range(segmentNum):
        # for segmentID in range(2):
            endPoint = (segmentID + 1) * sampleNumInTimeWindow
            partialEEG = allEEG[:endPoint]
            d, pval, chi2 = statisticalTester.ks_test(partialEEG)
            print('fileID = ' + fileID + ', segmentID = ' + str(segmentID)+ ', d = ' + str(d))
            dVec_L.append(d)
        dMat_L.append(dVec_L)
        fileNum += 1

        fh_dMat_L = open('../data/ks/dMat_L.pkl','wb')
        pickle.dump(dMat_L, fh_dMat_L)
        fh_dMat_L.close()

segmentNumsVec = np.array(segmentNums_L)
minSegmentNum = np.min(segmentNumsVec)
print('minSegmentNum = ' + str(minSegmentNum))
standardMiceNum = dMat_L[0][0].shape[0]
dTensor = np.zeros((fileNum, minSegmentNum, standardMiceNum))
fileID = 0
for dVec_L in dMat_L:
    segmentID = 0
    for d in dVec_L[:minSegmentNum]:
        print('dTensor.shape = ' + str(dTensor.shape) + ', d.shape = ' + str(d.shape))
        dTensor[fileID,segmentID,:] = np.array(d)
        print('dTensor[' + str(fileID) + ',' + str(segmentID) + ',:] = ' + str(dTensor[fileID,segmentID,:]))
        segmentID += 1
    fileID += 1

print('dTensor.shape = ' + str(dTensor.shape))
fh = open('../data/ks/dTensor.pkl','wb')
pickle.dump(dTensor, fh)
fh.close()
