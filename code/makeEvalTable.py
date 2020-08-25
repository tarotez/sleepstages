import sys
import pickle
import numpy as np
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

# print('$#$#$# measures_by_classifier_LL[0][0] =', measures_by_classifier_LL[0][0])
# print('$#$#$# measures_by_classifier_LL[0][-1] =', measures_by_classifier_LL[0][-1])
classifierNum = len(loaded_classifierIDs)
print('classifierNum =', classifierNum)

def printStats(vals, classifierNum):
    max_val = vals.max()
    mean_val = vals.mean()
    for classifierCnt in range(classifierNum):
        val = vals[classifierCnt]
        # if val == max_val and datasetType != 'test':
        #    print('{\\bf ' + '{:.3f}'.format(np.round(val,5)) + '}', end='')
        #else:
        #    print('{:.3f}'.format(np.round(val,5)), end='')
        print('{:.3f}'.format(np.round(val,5)), end='')
        if classifierCnt == classifierNum - 1:
            # prints the average if in the test mode
            # if datasetType == 'test':
            #    print(' & {:.3f}'.format(np.round(mean_val,5)), end='')
            # print('\\\\')
            print('')
        else:
            print(',  ', end='')

print('')
# if datasetType == 'test':
#     print(' Classifier & & 1 & 2 & 3 & Mean \\\\')
# else:
#    print('\multicolumn{2}{|c|}{} & CNN+ & CNN & Raw+ & Raw+ & Spec.+ & Raw & Spec. \\\\')
#    print('\multicolumn{2}{|c|}{} & LSTM & (All) & Spec. & Time & Time & & \\\\')

vals = np.zeros(classifierNum)
for measureIDIncrement in range(2):
    measureID = len(measureNames) + measureIDIncrement
    print('\\multicolumn{2}{|c|}{' + multiClassMeasureNames[measureIDIncrement] + '} & ', end='')
    vals = np.zeros(classifierNum)
    for classifierCnt in range(classifierNum):
        val = 0
        for groupID in range(groupNum):
            val += measures_by_classifier_LL[groupID][classifierCnt][measureID]
        val /= groupNum
        vals[classifierCnt] = val
    printStats(vals, classifierNum)

labelID = 0
for labelID in [1,0,2]:   # in the order of W, S, R
    for measureID in range(len(measureNames)):
        measureName = measureNames[measureID]
        if measureID == 0:
            print(stageLabels[labelID] + ', ' + measureName + ', ', end='')
        else:
            print('   ' + measureName + ', ', end='')
        # print('in for labelID, classifierNum =', classifierNum)
        vals = np.zeros(classifierNum)
        for classifierCnt in range(classifierNum):
            val = 0
            for groupID in range(groupNum):
                val += measures_by_classifier_LL[groupID][classifierCnt][measureID][labelID]
            val /= groupNum
            vals[classifierCnt] = val
        printStats(vals, classifierNum)

print('\n')
