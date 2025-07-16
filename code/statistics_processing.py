import numpy as np
import datetime
from parameterSetup import ParameterSetup

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

def connect_within_max_length(past_segment, segment, max_length):
    return np.r_[past_segment, segment] if len(past_segment) < max_length else past_segment

def standardize(segment, past_segment):
    if len(past_segment) > 1:
        if past_segment.std() > 0:
            # print('past_segment.mean() =', past_segment.mean(), ', past_segment.std() =', past_segment.std())
            standardized_segment = (segment - past_segment.mean()) / past_segment.std()
        else:
            standardized_segment = segment
    else:
        standardized_segment = segment
    # return standardized_segment, segment
    # return standardized_segment, np.r_[past_segment, segment]
    max_length = 500000
    return standardized_segment, connect_within_max_length(past_segment, segment, max_length)

def centralize(segment, past_segment):
    if len(past_segment) > 1:
        standardized_segment = segment - past_segment.mean()
    else:
        standardized_segment = segment
    # return standardized_segment, segment
    # return standardized_segment, np.r_[past_segment, segment]
    max_length = 500000
    return standardized_segment, connect_within_max_length(past_segment, segment, max_length)

class standardizer():
    def __init__(self, params, max_storage_length):
        # params = ParameterSetup()
        self.max_storage_length = max_storage_length
        self.connected = []
        self.segment_counter = 0
        self.early_trigger = params.standardization_early_trigger
        self.trigger_interval = params.standardization_trigger_interval
        self.mean = 0
        self.std = 0
        self.outlier_coef = 1000
        self.logDir = params.logDir

    def write_outlier_log(self, outlier):
        with open(self.logDir + '/standardization.log', 'a') as f:
            f.write('time = ' + str(datetime.datetime.now()) + ', segmentID = ' + str(self.segment_counter) + ', outlier =' + str(outlier) + '\n')
            f.flush()

    def remove_outliers(self, segment):
        assert type(segment) == list, 'the input segment to remove_outliers() is not a list'
        if self.segment_counter > 1 and self.std > 0:
            upper_bound = self.mean + self.outlier_coef * self.std
            lower_bound = self.mean - self.outlier_coef * self.std
            upper_std = self.mean + self.std
            lower_std = self.mean - self.std
            #------
            # write to the log directory when an outlier appears.
            if max(segment) > upper_bound:
                self.write_outlier_log(max(segment))
                segment = [upper_std if samplePoint > upper_bound else samplePoint for samplePoint in segment]

            if min(segment) < lower_bound:
                self.write_outlier_log(min(segment))
                segment = [lower_std if samplePoint < lower_bound else samplePoint for samplePoint in segment]

        return segment

    def standardize(self, new_samples):
        self.connected = self.connected + self.remove_outliers(list(new_samples))
        self.connected = self.connected[-self.max_storage_length:]
        if (self.segment_counter in self.early_trigger) or (self.segment_counter % self.trigger_interval) == 0:
            self.mean = np.mean(self.connected)
            self.std = np.std(self.connected)
        self.segment_counter += 1
        assert self.mean.shape == ()
        assert self.std.shape == ()
        if self.std > 0:
            return (new_samples - self.mean) / self.std
        else:
            return new_samples - self.mean

    def centralize(self, new_samples):
        self.connected = self.connected + self.remove_outliers(list(new_samples))
        self.connected = self.connected[-self.max_storage_length:]
        if (self.segment_counter in self.early_trigger) or (self.segment_counter % self.trigger_interval) == 0:
            self.mean = np.mean(self.connected)
        self.segment_counter += 1
        assert self.mean.shape == ()
        return new_samples - self.mean


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
