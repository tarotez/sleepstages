from __future__ import print_function
from functools import reduce
from operator import or_
import numpy as np
import pickle
from os import listdir
from os.path import splitext
import random
from classicalClassifier import ClassicalClassifier
from deepClassifier import DeepClassifier

def selectClassifierID(finalClassifierDir, classifier_type):
    # all_files = listdir(self.params.finalClassifierDir)
    classifierDict = {}
    with open(finalClassifierDir + '/classifierTypes.csv') as f:
        for line in f:
            elems = [elem.strip() for elem in line.split(',')]
            print('adding', elems[1], ':', elems[0], 'to dict.')
            classifierDict.update({elems[1] : elems[0]})
    classifierID = classifierDict[classifier_type]
    print('in selectClassifierID(), classifierType =', classifier_type, 'and classifierID =', classifierID)
    return classifierID
    # prefix = 'params.'
    # paramFiles = filterByPrefix(all_files, prefix)
    # paramFile = paramFiles[np.random.randint(len(paramFiles))]
    # return paramFile.split('.')[1]

#--------------------------------------------
# write names of files used for training the classifier
def writeTrainFileIDsUsedForTraining(params, classifier, fileTripletL):
    f = open(params.classifierDir + '/files_used_for_training.' + params.classifierID + '.csv', 'w')
    for trip in fileTripletL:
        f.write(str(trip[0]) + ',' + trip[1] + ',' + trip[2] + '\n')
    f.close()

def readTrainFileIDsUsedForTraining(params, classifierID):
    f = open(params.classifierDir + '/files_used_for_training.' + classifierID + '.csv', 'r')
    fileTripletL = [line.rstrip().split(',') for line in f]
    return fileTripletL   # = (eegAndStageFile, featureFile, fileID)

#--------------------------
# split samples into cross-validation subsets
def listSplit(lis, testNum, offset, randomize):
    testNum, offset = int(testNum), int(offset)
    if randomize:
        random.shuffle(lis)
    # print('len(lis) =', len(lis))
    # print('testNum =', testNum)
    # print('offset =', offset)
    return lis[:offset] + lis[(offset+testNum):], lis[offset:(offset+testNum)]

def crossValidationSplits(files, testNum):
    return map(lambda offset: listSplit(files, testNum, offset), range(0, len(files) - testNum + 1, testNum))

def leaveOneOut(files):
    return crossValidationSplits(files, 1)

#--------------------------
# recursively filter files using prefix
# def length(seq):
#    return len(seq)

# def car(seq):
#    return seq[0]

# def cdr(seq):
#    return seq[1:]

def filterByPrefix(seq, prefix):
    return list(filter(lambda x: x.startswith(prefix), seq))
    # return ([car(seq)] if car(seq).startswith(prefix) else []) + (filterByPrefix(cdr(seq), prefix) if length(seq) > 1 else [])

#--------------------------
# get EEG and feature files fulfilling special conditions specified by params.json
def filterEEGFiles(params, files):
    return filterByPrefix(files, params.eegFilePrefix)

def filterFeatureFiles(params, files):
    return filterByPrefix(files, params.featureFilePrefix)

def getFileIDFromEEGFile(fileName):
    return fileName.split('.')[1]

def getExtractorFromFeatureFile(fileName):
    return fileName.split('.')[1]

def getEMGLabelFromFeatureFile(fileName):
    return fileName.split('.')[2]

def getFileIDFromFeatureFile(fileName):
    return fileName.split('.')[3]

def getAllEEGFiles(params):
    eegAndStageFiles = filterEEGFiles(params, listdir(params.eegDir))
    if(len(eegAndStageFiles)) == 0:
        raise Exception('no eegAndStage...pkl file in', params.eegDir)
    return eegAndStageFiles

def fileIDsFromEEGFiles(eegFiles):
    return list(map(getFileIDFromEEGFile, eegFiles))

def foundInList(examined_string, targetValues):
    return reduce(lambda a, x: or_(a, examined_string == x), targetValues, False)

def filterFiles(originalL, parser, targetValues):
    return reduce(lambda a, x: a + [x] if foundInList(parser(x), targetValues) else a, originalL, [])
    # filtered = []
    # for elem in originalL:
    #    if foundInList(parser(elem), targetValues):
    #        filtered.append(elem)
    # return filtered

def excludeFiles(originalL, parser, targetValues):
    return reduce(lambda a, x: a + [x] if not foundInList(parser(x), targetValues) else a, originalL, [])

def getEEGFiles(params, fileIDs):
    f0 = filterEEGFiles(params, listdir(params.eegDir))
    return filterFiles(f0, getFileIDFromEEGFile, fileIDs)

def getFeatureFiles(params, fileIDs):
    f0 = filterFeatureFiles(params, listdir(params.featureDir))
    f1 = filterFiles(f0, getFileIDFromFeatureFile, fileIDs)
    f2 = filterFiles(f1, getExtractorFromFeatureFile, [params.extractorType])
    return filterFiles(f2, getEMGLabelFromFeatureFile, [params.label4withoutEMG])

def sortAndMerge(s1, s2, s3):
    return zip(sorted(s1), sorted(s2), sorted(s3))

def getEEGAndFeatureFiles(params, testNum, offset, randomize):
    eegAndStageFiles = getAllEEGFiles(params)
    train_eegAndStageFiles, test_eegAndStageFiles = listSplit(eegAndStageFiles, testNum, offset, randomize)
    # print('train_eegAndStageFiles =', train_eegAndStageFiles)
    train_fileIDs, test_fileIDs = list(map(fileIDsFromEEGFiles, (train_eegAndStageFiles, test_eegAndStageFiles)))
    train_featureFiles, test_featureFiles = list(map(lambda fileIDs: getFeatureFiles(params, fileIDs), (train_fileIDs, test_fileIDs)))
    train_fileTripletL = list(sortAndMerge(train_eegAndStageFiles, train_featureFiles, train_fileIDs))
    test_fileTripletL = list(sortAndMerge(test_eegAndStageFiles, test_featureFiles, test_fileIDs))
    return train_fileTripletL, test_fileTripletL

def fileIDs2triplets(params, train_fileIDs, test_fileIDs):
    train_eegAndStageFiles, test_eegAndStageFiles = list(map(lambda fileIDs: getEEGFiles(params, fileIDs), (train_fileIDs, test_fileIDs)))
    train_featureFiles, test_featureFiles = list(map(lambda fileIDs: getFeatureFiles(params, fileIDs), (train_fileIDs, test_fileIDs)))
    train_fileTripletL = list(sortAndMerge(train_eegAndStageFiles, train_featureFiles, train_fileIDs))
    test_fileTripletL = list(sortAndMerge(test_eegAndStageFiles, test_featureFiles, test_fileIDs))
    return train_fileTripletL, test_fileTripletL

def getEEGAndFeatureFilesByClassifierID(params,classifierIDforTrainingFiles):
    train_fileTripletL = readTrainFileIDsUsedForTraining(params, classifierIDforTrainingFiles)
    train_fileIDs = [train_fileID for _, _, train_fileID in train_fileTripletL]
    all_eegAndStageFiles = getAllEEGFiles(params)
    all_fileIDs = fileIDsFromEEGFiles(all_eegAndStageFiles)
    test_fileIDs = excludeFiles(all_fileIDs, lambda x: x, train_fileIDs)
    return fileIDs2triplets(params, train_fileIDs, test_fileIDs)

def getEEGAndFeatureFilesByExcludingFromTrainingByPrefix(params, test_file_prefix):
    all_eegAndStageFiles = getAllEEGFiles(params)
    all_fileIDs = fileIDsFromEEGFiles(all_eegAndStageFiles)
    train_fileIDs = reduce(lambda a, x: a + [x] if not x.startswith(test_file_prefix) else a, all_fileIDs, [])
    test_fileIDs = reduce(lambda a, x: a + [x] if x.startswith(test_file_prefix) else a, all_fileIDs, [])
    [print('included ', fileID) for fileID in train_fileIDs]
    [print('excluded ', fileID) for fileID in test_fileIDs]
    return fileIDs2triplets(params, train_fileIDs, test_fileIDs)

def getEEGAndFeatureFilesByExcludingFromTrainingByMouseIDs(params, mouseIDs):
    all_eegAndStageFiles = getAllEEGFiles(params)
    all_fileIDs = fileIDsFromEEGFiles(all_eegAndStageFiles)
    print('mouseIDs =', mouseIDs)
    def contains(targetstr, patterns):
        # print('targetstr =', targetstr)
        for pattern in patterns:
            # print('  pattern =', pattern, '')
            if targetstr.find(pattern) >= 0:
                # print('  matched!!')
                return True
        # print('no match.')
        return False
    train_fileIDs = reduce(lambda a, x: a + [x] if not contains(x, mouseIDs) else a, all_fileIDs, [])
    test_fileIDs = reduce(lambda a, x: a + [x] if contains(x, mouseIDs) else a, all_fileIDs, [])
    [print('included ', fileID) for fileID in train_fileIDs]
    [print('excluded ', fileID) for fileID in test_fileIDs]
    return fileIDs2triplets(params, train_fileIDs, test_fileIDs)

def getFilesNotUsedInTrain(params, train_fileIDs):
    eegAndStageFiles = getAllEEGFiles(params)
    # print('after eegAndStageFiles = getAllEEGFiles(params):')
    # [print('fileName =', fileName) for fileName in eegAndStageFiles]
    all_fileIDs = fileIDsFromEEGFiles(eegAndStageFiles)
    # print('after all_fileIDs = fileIDsFromEEGFiles(eegAndStageFiles):')
    # [print('fileID =', fileID) for fileID in all_fileIDs]
    test_fileIDs = excludeFiles(all_fileIDs, lambda x: x, train_fileIDs)
    # [print('test_fileID =', test_fileID) for test_fileID in test_fileIDs]
    test_eegAndStageFiles = getEEGFiles(params, test_fileIDs)
    test_featureFiles = getFeatureFiles(params, test_fileIDs)
    test_fileTripletL = list(sortAndMerge(test_eegAndStageFiles, test_featureFiles, test_fileIDs))
    return test_fileTripletL

#----------------------------------------
# choose a classifier based on parameters
def findClassifier(params, paramID, classLabels):
    classifierID = params.classifierID   # classifierID is generated each time ParameterSetup is called
    classifierType = params.classifierType
    classifierParams = params.classifierParams
    # for reading data
    print('classifierID = ' + classifierID)
    print('classifierType = ' + classifierType)
    print('for testing, use:')
    print('python computeTestError.py ' + classifierID)
    # print('for visualization, use:')
    # print('tensorboard --logdir ../data/deep_params/' + excludedFileID + '/' + classifierID)
    if classifierType == 'deep':
        print('in fileManagement.finClassifier(), params.networkType =', params.networkType)
        print('classLabels =', classLabels)
        classifier = DeepClassifier(classLabels, classifierID=classifierID, paramsForDirectorySetup=params, paramsForNetworkStructure=params)
        # paramsForDirectorySetup = ParameterSetup()
        ### paramsForNetworkStructure = ParameterSetup(paramFileName='paramsForNetwork.json')
        # paramsForNetworkStructure = ParameterSetup()
        # classifier = DeepClassifier(classLabels, excludedFileID=excludedFileID, classifierID=classifierID, paramsForDirectorySetup=paramsForDirectorySetup, paramsForNetworkStructure=paramsForNetworkStructure)
    else:
        classifier = ClassicalClassifier(params, paramID)
    return classifier

#------------------
# functions below are bit old

def readStandardMice(params):
    standardMiceDir = params.standardMiceDir
    standardMice_L = []
    files_L = listdir(standardMiceDir)
    for fileFullName in files_L:
        fileIDwithPrefix, file_extension = splitext(fileFullName)
        if file_extension == '.pkl':
            dataFileHandler = open(standardMiceDir + '/' + fileFullName, 'rb')
            if fileFullName.startswith('eegAndStage.'):
                (eeg, emg, stageSeq, timeStamps) = pickle.load(dataFileHandler)
            elif fileFullName.startswith('eegOnly.'):
                (eeg, emg, timeStamps) = pickle.load(dataFileHandler)
            dataFileHandler.close()
            standardMice_L.append(eeg)
    return np.array(standardMice_L), files_L

def readdMat(params):
    try:
        ksDir = params.ksDir
        fh = open(ksDir + '/dMat.pkl', 'rb')
        dMat = pickle.load(fh)
        fh.close()
        return dMat
    except EnvironmentError:
        return np.zeros((56, 5))

def readdTensor(params):
    try:
        ksDir = params.ksDir
        fh = open(ksDir + '/dTensor.pkl', 'rb')
        dTensor = pickle.load(fh)
        fh.close()
        return dTensor
    except EnvironmentError:
        return np.zeros((56, 60, 5))

def getFileIDs(targetDir, prefix):
    files_L = listdir(targetDir)
    fileIDs = []
    for fileFullName in files_L:
        fileIDwithPrefix, file_extension = splitext(fileFullName)
        if file_extension == '.pkl' and fileIDwithPrefix.startswith(prefix):
            elems = fileIDwithPrefix.split('.')
            fileID = elems[1]
            fileIDs.append(fileID)
    # print('fileIDs = ' + str(fileIDs))
    return fileIDs

def getFileIDsCorrespondingToClassifiers(params, targetDir, prefix):
    fileIDpairs = []
    for fileFullName in listdir(targetDir):
        # print('reading ' + fileFullName + ':')
        fileIDwithPrefix, file_extension = splitext(fileFullName)
        longPrefix = prefix + '.' + params.classifierType
        if params.useEMG:
            longPrefix = longPrefix + '.withEMG'
        else:
            longPrefix = longPrefix + '.withoutEMG'
        # print('  longPrefix = ' + longPrefix)
        if file_extension == '.pkl' and fileIDwithPrefix.startswith(longPrefix):
            # print('   getting fileID from ' + fileIDwithPrefix)
            elems = fileIDwithPrefix.split('.')
            excludedFileID = elems[4]
            classifierID = elems[6]
            fileIDpairs.append([excludedFileID, classifierID])
    return fileIDpairs
