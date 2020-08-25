import pickle

def loadStatistics(pickledDir, classifierID):
    f = open(pickledDir + '/test_result.' + classifierID + '.pkl', 'rb')
    test_result = pickle.load(f)
    f.close()
    return test_result
