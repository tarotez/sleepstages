from __future__ import print_function
import sys
from os import listdir
import numpy as np
from parameterSetup import ParameterSetup
from fileManagement import getFileIDs
from algorithmFactory import AlgorithmFactory
from outlierMouseFilter import OutlierMouseFilter

#---------------
# set up parameters
args = sys.argv
if len(args) > 1:
    option = args[1]
else:
    option = ''

# get params shared by programs
params = ParameterSetup()
useEMG = params.useEMG
eegDir = params.eegDir
featureDir = params.featureDir
pastStageLookUpNum = params.pastStageLookUpNum

extractorType = params.extractorType
factory = AlgorithmFactory(extractorType)
extractor = factory.generateExtractor()

oFilter = OutlierMouseFilter()

#---------------
# print out parameter setting for feature extraction

if useEMG:
    print('using EMG.')
    label4EMG = params.label4withEMG
else:
    label4EMG = params.label4withoutEMG
    print('not using EMG')

print('pastStageLookUpNum = ' + str(pastStageLookUpNum))

#---------------
# read pickled files that stores EEG and stage labels

prefix = 'eegAndStage'

fileIDs = getFileIDs(eegDir, prefix)
### fileIDs = ['HET-NR-D0717', 'DBL-NO-D1473', 'HET-NO-D0905']

for fileID in fileIDs:
    print('fileID = ' + str(fileID))
    featureFileName = params.featureFilePrefix + '.' + params.extractorType + '.' + label4EMG + '.' + fileID + '.pkl'
    if oFilter.isOutlier(fileID):
        print('file ' + fileID + ' is an outlier, so skipping.')
    else:
        flag4extraction = 1
        if option != '-o':
            for fileName in listdir(featureDir):
                if fileName == featureFileName:
                    flag4extraction = 0
                    print(featureFileName + ' already exists, so skipping')
                    break
        if flag4extraction:
            extractor.featureExtraction(params, fileID)
