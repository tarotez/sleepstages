import numpy as np

def linearFit(y):
    x = np.array([i for i, _ in enumerate(y)])
    X = np.c_[x, np.ones(len(x))]
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose()), y)
    # z = np.matmul(beta, X.transpose())
    return beta

def extrapolate(y):
    beta = linearFit(y)
    new_x = np.array([i + len(y) for i, _ in enumerate(y)])
    new_X = np.c_[new_x, np.ones(len(new_x))]
    new_y = np.matmul(beta, new_X.transpose())
    return new_y

def subtractLinearFit(segment, previous_segment, sampleID):
    # subtracted_segment =
    if len(previous_segment) > 1:
        extrapolation = extrapolate(previous_segment)
        subtracted_segment = segment[:sampleID] - extrapolation[:sampleID]
    else:
        subtracted_segment = segment
    # return standardized_segment, np.r_[past_segment, segment]
    concatenated_segment = np.r_[subtracted_segment, np.zeros((len(segment) - len(subtracted_segment)))]
    return concatenated_segment, segment

def standardize(segment, past_segment):
    if len(past_segment) > 1:
        if past_segment.std() > 0:
            # print('past_segment.mean() =', past_segment.mean(), ', past_segment.std() =', past_segment.std())
            standardized_segment = (segment - past_segment.mean()) / past_segment.std()
        else:
            standardized_segment = segment
    else:
        standardized_segment = segment
    # return standardized_segment, np.r_[past_segment, segment]
    return standardized_segment, segment

def centralize(segment, past_segment):
    if len(past_segment) > 1:
        standardized_segment = segment - past_segment.mean()
    else:
        standardized_segment = segment
    # return standardized_segment, np.r_[past_segment, segment]
    return standardized_segment, segment

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
