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

fileIDs = getFileIDs(pickledDir, eegFilePrefix)
### fileIDs = ['HET-NR-D0717', 'DBL-NO-D1473', 'HET-NO-D0905']

dMat_L = []
segmentNums_L = []
fileNum = 0
for fileID in fileIDs:
    fileFullName = eegFilePrefix + '.' + fileID + '.pkl'
    useThisMouse = True
    for standardMiceFileFullName in standardMiceFileFullNames:
        if fileFullName.startswith(standardMiceFileFullName):
            useThisMouse = False
    if useThisMouse:
        dataFileHandler = open(pickledDir + '/' + fileFullName, 'rb')
        (eeg, emg, stageSeq, timeStamps) = pickle.load(dataFileHandler)
        allEEG = eeg
        # dVec_L = []
        segmentNum = np.int(np.ceil(allEEG.shape[0]/sampleNumInTimeWindow))
        segmentNums_L.append(segmentNum)
        ### for segmentID in range(segmentNum):
        ### endPoint = segmentID * sampleNumInTimeWindow
        ### endPoint = segmentID * sampleNumInTimeWindow
        ### partialEEG = allEEG[:endPoint]
        ### d, pval, chi2 = statisticalTester.ks_test(partialEEG)
        d, pval, chi2 = statisticalTester.ks_test(allEEG)
        # print('fileID = ' + fileID + ', segmentID = ' + str(segmentID)+ ', d = ' + str(d))
        print('fileID = ' + fileID + ', segmentNum = ' + str(segmentNum)+ ', d = ' + str(d))
        #### dVec_L.append(np.mean(d))
        # dVec_L.append(d)
        dMat_L.append(d)
        fileNum += 1

# segmentNumsVec = np.array(segmentNums_L)
# minSegmentNum = np.min(segmentNumsVec)

standardMiceNum = len(dMat_L[0])
dMat = np.zeros((fileNum, standardMiceNum))
fileID = 0
for d in dMat_L:
    # dMat[fileID,:] = np.array(dVec_L[:minSegmentNum])
    dMat[fileID,:] = d
    print('dMat[' + str(fileID) + '] = ' + str(dMat[fileID,:]))
    fileID += 1

print('dMat.shape = ' + str(dMat.shape))
fh = open('../data/ks/dMat.pkl','wb')
pickle.dump(dMat, fh)
fh.close()
