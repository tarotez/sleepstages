from __future__ import print_function
import sys
from os import listdir
from os.path import splitext
import numpy as np
from parameterSetup import ParameterSetup
from featureconnector import Featureconnector

#---------------
# set up parameters
args = sys.argv
if len(args) > 1:
    option = args[1]
else:
    option = ''
    print('extractorType is not specified. Usage example: python connectFeatures.py wavelet')
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
filterName = 'connected'
originalPrefix = featureFilePrefix + '.' + extractorType + '.' + label4EMG
outputPrefix = featureFilePrefix + '.' + extractorType + '-' + filterName + '.' + label4EMG
connector = FeatureConnector()

#---------------
# read files and filter
for excludeFileName in listdir(featureDir):
    for inputFileName in listdir(featureDir):
        if inputFileName.startswith(originalPrefix) and not inputFileName.startswith(outputPrefix):
            if inputFileName != excludeFileName:
                fileIDwithPrefix, file_extension = splitext(inputFileName)
                elems = fileIDwithPrefix.split('.')
                # print('elems = ' + str(elems))
                fileID = elems[3]
                outputFileName = outputPrefix + '.' + fileID + '.pkl'
                flag4extraction = 1
                if flag4extraction:
                    inputFeaturePath = featureDir + '/' + inputFileName
                    outputFeaturePath = featureDir + '/' + outputFileName
                    connector.featureConnecting(inputFeaturePath)

connector.saveOutputTensor(outputFeaturePath)
