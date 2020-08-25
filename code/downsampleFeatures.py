from __future__ import print_function
import sys
from os import listdir
from os.path import splitext
import numpy as np
from parameterSetup import ParameterSetup
from featureDownSampler import FeatureDownSampler
#---------------
# set up parameters
args = sys.argv
if len(args) > 1:
    option = args[1]
else:
    option = ''
    print('extractorType is not specified. Usage example: python downSampleFeatures.py wavelet')

extractorType = option

#---------------
# get params shared by programs
params = ParameterSetup()
featureDir = params.featureDir
featureFilePrefix = params.featureFilePrefix
useEMG = params.useEMG
if useEMG:
    label4EMG = params.label4withEMG
else:
    label4EMG = params.label4withoutEMG
filterName = 'downsampled'
originalPrefix = featureFilePrefix + '.' + extractorType + '.' + label4EMG
outputPrefix = featureFilePrefix + '.' + extractorType + '-' + filterName + '.' + label4EMG

downSampler = FeatureDownSampler()
outputDim = 320

#---------------
# read files and filter
for inputFileName in listdir(featureDir):
    if inputFileName.startswith(originalPrefix) and not inputFileName.startswith(outputPrefix):
        fileIDwithPrefix, file_extension = splitext(inputFileName)
        elems = fileIDwithPrefix.split('.')
        # print('elems = ' + str(elems))
        fileID = elems[3]
        outputFileName = outputPrefix + '.' + fileID + '.pkl'
        flag4extraction = 1
        '''
        if option != '-o':
            for fileName in listdir(featureDir):
                if fileName == outputFileName:
                    print(outputFileName + ' already exists, so skipping')
                    flag4extraction = 0
                    break
        '''
        if flag4extraction:
            inputFeaturePath = featureDir + '/' + inputFileName
            outputFeaturePath = featureDir + '/' + outputFileName
            downSampler.featureDownSampling(inputFeaturePath, outputFeaturePath, outputDim)
