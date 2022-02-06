from __future__ import print_function
import sys
from os import listdir
import pickle
import string
import random
from functools import reduce
from parameterSetup import ParameterSetup
from classifierTrainer import connectSamplesAndTrain
from fileManagement import fileIDs2triplets
from tester import test_by_classifierID

def read_blocks(params, splitID):
    with open(params.pickledDir + '/blocks_of_records.' + splitID + '.csv') as f:
    # with open(params.pickledDir + '/blocks_of_records.csv') as f:
        return [line.rstrip().split(',') for line in f]

args = sys.argv
splitID = args[1]
print('splitID =', splitID)
crossValidationID = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
print('crossValidationID =', crossValidationID)
datasetType = 'test'
paramDir = '../data/compare'
outputDir = '../data/pickled'
classifierIDsByMethod = []
paramFiles = listdir(paramDir)
list.sort(paramFiles)
for paramFileName in paramFiles:
    if paramFileName.startswith('params.') and not paramFileName.endswith('~'):
        print('paramFileName =', paramFileName)
        params = ParameterSetup(paramDir, paramFileName, outputDir)

        def stage_restriction():
            return lambda x: restrictStages(params, x, params.maximumStageNum)

        print('splitID =', splitID)
        blocks = read_blocks(params, splitID)
        classifierIDsByBlock = []
        for foldID, test_fileIDs in enumerate(blocks):
            print('')
            print('foldID =', foldID)
            train_fileIDs = reduce(lambda a, x: a + x[1] if x[0] != foldID else a, enumerate(blocks), [])
            print('train_fileIDs =', train_fileIDs)
            print('test_fileIDs =', test_fileIDs)
            train_fileTripletL, test_fileTripletL = fileIDs2triplets(params, train_fileIDs, test_fileIDs)
            print('%%%W len(train_fileTripletL) =', len(train_fileTripletL))
            print('%%%W len(test_fileTripletL) =', len(test_fileTripletL))
            if len(train_fileTripletL) > 0:
                # print('training by', train_fileTripletL)
                # print('')
                classifierID = connectSamplesAndTrain(params, train_fileTripletL, stage_restriction)
                classifierIDsByBlock.append(classifierID)
            else:
                print('No file for training.')
        classifierIDsByMethod.append(classifierIDsByBlock)
        with open(outputDir + '/crossvalidation_metadata.' + crossValidationID + '.pkl', 'wb') as f:
            pickle.dump((splitID, classifierIDsByMethod), f)

print('crossValidationID =', crossValidationID)
