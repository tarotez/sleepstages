import sys
import pickle
from parameterSetup import ParameterSetup
from tester import printMetadata, test_by_classifierID

args = sys.argv

if args[1] == 'test':
    datasetType = 'test'
    classifierIDs = args[2:]
elif args[1] == 'validation':
    datasetType = 'validation'
    classifierIDs = args[2:]
else:
    datasetType = 'validation'
    classifierIDs = args[1:]

params = ParameterSetup()
printMetadata(params)
pickledDir = params.pickledDir
# sensitivity_by_classifier_L, specificity_by_classifier_L, accuracy_by_classifier_L, precision_by_classifier_L, measures_by_classifier_L = [], [], [], [], []
measures_by_classifier_L = []
for classifierID in classifierIDs:
    measures_by_classifier_L.append(test_by_classifierID(params, datasetType, classifierID))

if datasetType == 'test':
    f = open(pickledDir + '/test_result.measures.' + classifierIDs[0] + '.etc.test.pkl', 'wb')
else:
    f = open(pickledDir + '/test_result.measures.' + classifierIDs[0] + '.etc.pkl', 'wb')
pickle.dump((classifierIDs, measures_by_classifier_L), f)
f.close()












#####
