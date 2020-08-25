import sys
import numpy as np
from parameterSetup import ParameterSetup
from evaluationCriteria import y2sensitivity, y2confusionMat, printConfusionMat, mathewsCorrelationCoefficient, multiClassMCC

args = sys.argv
test_path = args[1]
pred_path = args[2]

params = ParameterSetup()
stageLabels = params.stageLabels4evaluation
labelNum = len(stageLabels)
print('skip line num:', params.metaDataLineNumUpperBound4stage)
test_fp = open('../data/' + test_path)
for i in range(params.metaDataLineNumUpperBound4stage):    # skip lines that describes metadata
    line = test_fp.readline()
    if line.startswith(',,,,,'):
        break
    if i == params.metaDataLineNumUpperBound4stage - 1:
        print('metadata (header) for the EEG file is not correct.')
        quit()

y_test = []
for line in test_fp:
    y_test.append(line.rstrip().split(',')[2])

finalEpoch = 2160
y_test = np.array(['S' if elem == 'NR' else elem for elem in y_test])[1:finalEpoch+1]
y_pred = np.array(['S' if line.rstrip() == '1' else line.rstrip() for line in open('../data/' + pred_path)])[:finalEpoch]

print('len(y_test) =', len(y_test))
print('len(y_pred) =', len(y_pred))
print('y_test[:30] =', y_test[:100])
print('y_pred[:30] =', y_pred[:100])

(stageLabels, sensitivity, specificity, accuracy) = y2sensitivity(y_test, y_pred)
(stageLabels4confusionMat, confusionMat) = y2confusionMat(y_test, y_pred)
printConfusionMat(stageLabels4confusionMat, confusionMat)

y_matching = (y_test == y_pred)
print('y_matching =', y_matching)
correctNum = sum(y_matching)
print('correctNum =', correctNum)
y_length = y_pred.shape[0]
precision = correctNum / y_length
print('labelNum = ' + str(labelNum))
for labelID in range(labelNum):
    print('  stageLabel = ' + stageLabels[labelID] + ', sensitivity = ' + "{0:.3f}".format(sensitivity[labelID]) + ', specificity = ' + "{0:.3f}".format(specificity[labelID]) + ', accuracy = ' + "{0:.3f}".format(accuracy[labelID]))
print('  precision = ' + "{0:.5f}".format(precision) + ' (= ' + str(correctNum) + '/' + str(y_length) +')')

for targetStage in ['W','S','R']:
    mcc = mathewsCorrelationCoefficient(stageLabels4confusionMat, confusionMat, targetStage)
    print('  mcc for ' + targetStage + ' = ' + "{0:.5f}".format(mcc))
mcMCC = multiClassMCC(stageLabels4confusionMat, confusionMat)
print('  multi class mcc = ' + "{0:.5f}".format(mcc))
print('')
