import numpy as np
import sys
from parameterSetup import ParameterSetup
from evaluationCriteria import y2sensitivity, y2confusionMat, printConfusionMat, mathewsCorrelationCoefficient, multiClassMCC

args = sys.argv
params = ParameterSetup()
predFileName = args[1]
testFilePath = args[2]

y_pred = []
with open(params.predDir + '/' + predFileName, 'r') as predFile:
    for line in predFile:
        p = line.rstrip()
        p = 'S' if p == '1' else p
        # print('pred:', p)
        y_pred.append(p)

y_test = []
with open(testFilePath) as testFile:
    for i in range(params.metaDataLineNumUpperBound4stage):    # skip lines that describes metadata
        line = testFile.readline()
        if line.startswith(params.cueWhereStageDataStarts):
            break
        if i == params.metaDataLineNumUpperBound4stage - 1:
            print('metadata header for stage file was not correct.')
            quit()
    for line in testFile:
        elems = line.split(',')
        if len(elems) > 2:
            t = elems[2]
            # print('test:', t)
            t = 'W' if t == 'RW' else t
            y_test.append(t)

y_pred = np.array(y_pred[11:])
y_test = np.array(y_test[11:])

# print('y_pred =', y_pred)
# print('y_test =', y_test)

(stageLabels, sensitivity, specificity, accuracy, precision, f1score) = y2sensitivity(y_test, y_pred)
(stageLabels4confusionMat, confusionMat) = y2confusionMat(y_test, y_pred, params.stageLabels4evaluation)
printConfusionMat(stageLabels4confusionMat, confusionMat)
print('sensitivity =', sensitivity)
print('specificity =', specificity)
print('accuracy =', accuracy)
print('precision =', precision)
