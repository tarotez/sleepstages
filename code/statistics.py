import numpy as np

def standardize(segment, allPast):

    if len(allPast) > 1:
        if allPast.std() > 0:
            # print('allPast.mean() =', allPast.mean(), ', allPast.std() =', allPast.std())
            standardized_segment = (segment - allPast.mean()) / allPast.std()
        else:
            standardized_segment = segment
    else:
        standardized_segment = segment
    return standardized_segment, np.r_[allPast, segment]

'''
def recompMean(newVector, oldMean, oldSampleNum):
    newMean = (oldMean * oldSampleNum + np.sum(newVector)) / (oldSampleNum + newVector.shape[0])
    # print('oldMean = ' + str(oldMean) + ', oldSampleNum = ' + str(oldSampleNum) + ', np.sum(newVector) = ' + str(np.sum(newVector)) + ', newVector.shape[0] = ' + str(newVector.shape[0]) + ', newMean = ' + str(newMean))
    return newMean

def recompVariance(newVector, oldVar, oldMean, newMean, oldSampleNum):
    if oldSampleNum == 0:
        return np.var(newVector)
    else:
        return (oldSampleNum * (oldVar + (newMean - oldMean) ** 2) + np.sum((newVector - newMean) ** 2)) / (oldSampleNum + newVector.shape[0])
'''
