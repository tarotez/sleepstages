import sys
import pickle
import numpy as np
import math
from parameterSetup import ParameterSetup

#--------------
# print in the format of a latex table
params = ParameterSetup()
pickledDir = params.pickledDir
paramDir = params.pickledDir

args = sys.argv

if args[1] == 'test':
    datasetType = 'test'
    groupIDs = args[2:]
elif args[1] == 'validation':
    datasetType = 'validation'
    groupIDs = args[2:]
else:
    datasetType = 'validation'
    groupIDs = args[1:]
groupNum = len(groupIDs)
measures_by_classifier_LL = []
for groupID in groupIDs:
    if datasetType == 'test':
        f = open(pickledDir + '/test_result.measures.' + groupID + '.etc.test.pkl', 'rb')
    else:
        f = open(pickledDir + '/test_result.measures.' + groupID + '.etc.pkl', 'rb')
    loaded_classifierIDs, measures_by_classifier_L = pickle.load(f)
    print('for groupID =', groupID, ', loaded_classifierIDs =', loaded_classifierIDs)
    print('len(measures_by_classifier_L) =', len(measures_by_classifier_L))
    measures_by_classifier_LL.append(measures_by_classifier_L)
    f.close()

stageLabels = ['S', 'W', 'R']
measureNames = ['sensitivity', 'specificity', 'accuracy   ', 'precision  ', 'F1 score   ', 'MCC        ']
multiClassMeasureNames = ['multiclass MCC', 'multiclass accuracy']
# methodOrigOrder = ['UN', 'UL', 'LR', 'RF', 'AB', 'Rw+Sp', 'Rw+ZT', 'Sp+ZT', 'Rw', 'Sp']
methodNames = ['UN', 'UL', 'LR', 'RF', 'AB', 'Rw', 'Sp', 'Rw+Sp', 'Rw+ZT', 'Sp+ZT']
# methodOrder = [0, 1, 2, 3, 4, 8, 9, 5, 6, 7]
methodOrder = list(range(len(methodNames)))

# print('$#$#$# measures_by_classifier_LL[0][0] =', measures_by_classifier_LL[0][0])
# print('$#$#$# measures_by_classifier_LL[0][-1] =', measures_by_classifier_LL[0][-1])
methodNum = len(loaded_classifierIDs)
print('loaded_classifierIDs =', loaded_classifierIDs)
print('methodNum =', methodNum)

def printRawStats(f, valL, methodNum):
    # print('methodNum =', methodNum)
    # max_val = vals.max()
    # mean_val = vals.mean()
    # print('starting printRawStats():')
    for methodID in range(methodNum):
        byGroup = valL[methodID]
        # if val == max_val and datasetType != 'test':
        #    print('{\\bf ' + '{:.3f}'.format(np.round(val,5)) + '}', end='')
        #else:
        #    print('{:.3f}'.format(np.round(val,5)), end='')
        # print('')
        # print('  byGroup:')
        # print('    ', end='')
        # print(byGroup)
        f.write(', '.join([str(elem) for elem in byGroup]))
        if methodID == methodNum - 1:
            # prints the average if in the test mode
            # if datasetType == 'test':
            #    print(' & {:.3f}'.format(np.round(mean_val,5)), end='')
            # print('\\\\')
            f.write('\n')
        else:
            f.write(', ')

# print('')
# if datasetType == 'test':
#     print(' Classifier & & 1 & 2 & 3 & Mean \\\\')
# else:
#    print('\multicolumn{2}{|c|}{} & CNN+ & CNN & Raw+ & Raw+ & Spec.+ & Raw & Spec. \\\\')
#    print('\multicolumn{2}{|c|}{} & LSTM & (All) & Spec. & Time & Time & & \\\\')

with open('../data/ms/table1/validate_by_classifiers.csv', 'w') as f:

    # make a list of attributes
    f.write(' , , ')
    attrL = []
    for methodID in methodOrder:
        byGroup = []
        # print('groupNum =', groupNum)
        for groupID in range(groupNum):
            # print('measures_by_classifier_LL[groupID][methodID][measureID] =', measures_by_classifier_LL[groupID][methodID][measureID])
            byGroup.append(methodNames[methodID] + ' validation ' + str(groupID+1))
        attrL.append(byGroup)
    printRawStats(f, attrL, methodNum)

    # make a list of labels
    for measureIDIncrement in [1,0]:
        measureID = len(measureNames) + measureIDIncrement
        # print('\\multicolumn{2}{|c|}{' + multiClassMeasureNames[measureIDIncrement] + '} & ', end='')
        f.write('all, ' + multiClassMeasureNames[measureIDIncrement] + ', ')
        valL = []
        for methodID in methodOrder:
            byGroup = []
            # print('groupNum =', groupNum)
            for groupID in range(groupNum):
                # print('measures_by_classifier_LL[groupID][methodID][measureID] =', measures_by_classifier_LL[groupID][methodID][measureID])
                byGroup.append(measures_by_classifier_LL[groupID][methodID][measureID])
            valL.append(byGroup)
        # print('valL:')
        # print(valL)
        # print('')
        printRawStats(f, valL, methodNum)

    labelID = 0
    for labelID in [1,0,2]:   # in the order of W, S, R
        for measureID, measureName in enumerate(measureNames):
            if measureID == 0:
                f.write(stageLabels[labelID] + ', ' + measureName + ', ')
            else:
                f.write(' , ' + measureName + ', ')
            # print('in for labelID, methodNum =', methodNum)
            valL = []
            for methodID in methodOrder:
                byGroup = []
                # print('groupNum =', groupNum)
                for groupID in range(groupNum):
                    # print('measures_by_classifier_LL[groupID][methodID][measureID] =', measures_by_classifier_LL[groupID][methodID][measureID])
                    byGroup.append(measures_by_classifier_LL[groupID][methodID][measureID][labelID])
                valL.append(byGroup)
            # print('valL:')
            # print(valL)
            # print('')
            printRawStats(f, valL, methodNum)

with open('../data/ms/table1/validate_by_classifiers.csv') as f:
    for line in list(f)[1:]:
        vals = line.split(',')[2:]
        for i in range(methodNum):
            print(format(math.floor(np.mean(np.array([float(elem) for elem in vals[i*3:(i+1)*3]])) * 1000) / 10), end='')
            if i == methodNum - 1:
                print('')
            else:
                print(',', end='')
