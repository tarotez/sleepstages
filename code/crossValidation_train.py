from __future__ import print_function
import sys
from os import listdir
import pickle
import string
import random
from parameterSetup import ParameterSetup
from classifierTrainer import connectSamplesAndTrain
from fileManagement import getEEGAndFeatureFiles, getEEGAndFeatureFilesByExcludingFromTrainingByMouseIDs
from tester import test_by_classifierID

def read_blocks(params, splitID):
    with open(params.pickledDir + '/blocks_of_records.' + splitID + '.csv') as f:
    # with open(params.pickledDir + '/blocks_of_records.csv') as f:
        return [line.split(',') for line in f]

args = sys.argv
splitID = args[1]
print('splitID =', splitID)
crossValidationID = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
print('crossValidationID =', crossValidationID)
datasetType = 'test'
paramDir = '../data/compare'
outputDir = '../data/pickled'
classifierIDsByParam = []
paramFiles = listdir(paramDir)
list.sort(paramFiles)
for paramFileName in paramFiles:
    if paramFileName.startswith('params.') and not paramFileName.endswith('~'):
        print('paramFileName =', paramFileName)
        params = ParameterSetup(paramDir, paramFileName, outputDir)
        print('splitID =', splitID)
        classifierIDsByBlock = []
        for foldID, mouseIDs in enumerate(read_blocks(params, splitID)):
            print('')
            print('foldID =', foldID)
            # print('mouseIDs =', mouseIDs)
            train_fileTripletL, test_fileTripletL = getEEGAndFeatureFilesByExcludingFromTrainingByMouseIDs(params, mouseIDs)
            print('len(train_fileTripletL) =', len(train_fileTripletL))
            if len(train_fileTripletL) > 0:
                # print('training by', train_fileTripletL)
                # print('')
                classifierID = connectSamplesAndTrain(params, train_fileTripletL)
                classifierIDsByBlock.append(classifierID)
            else:
                print('No file for training.')
        classifierIDsByParam.append(classifierIDsByBlock)
        with open(outputDir + '/crossvalidation_metadata.' + crossValidationID + '.pkl', 'wb') as f:
            pickle.dump((splitID, classifierIDsByParam), f)

print('crossValidationID =', crossValidationID)
