import numpy as np
from itertools import groupby
import operator
from parameterSetup import ParameterSetup

def up_or_down_sampling(signal_rawarray, model_samplePointNum, observed_samplePointNum):
    # downsampling
    # print('-------')
    # print('model_samplePointNum =', model_samplePointNum)
    # print('observed_samplePointNum =', observed_samplePointNum)
    if model_samplePointNum < observed_samplePointNum:
        print('-> downsampling')
        print('before downsampling: signal_rawarray.shape =', signal_rawarray.shape)
        epochNum = max(1, int(np.floor(1.0 * signal_rawarray.shape[0] / observed_samplePointNum)))
        print('epochNum =', epochNum)
        multiple = int(np.floor(1.0 * signal_rawarray.shape[0] / model_samplePointNum)) * model_samplePointNum * epochNum
        split_signal = np.array_split(signal_rawarray[:multiple], model_samplePointNum * epochNum)
        # for seg in split_signal:
        #     print('len(seg) =', len(seg))
        signal_rawarray = np.array([seg.mean() for seg in split_signal])
        # print('len(split_signal) =', len(split_signal))
        # print('split_signal[0].shape =', split_signal[0].shape)
        # print('after downsampling: signal_rawarray.shape =', signal_rawarray.shape)

    # upsampling
    if model_samplePointNum > observed_samplePointNum:
        upsample_rate = np.int(np.ceil(1.0 * model_samplePointNum / observed_samplePointNum))
        signal_rawarray = np.array([[elem] * upsample_rate for elem in signal_rawarray]).flatten()[:model_samplePointNum]

    return signal_rawarray

def supersample(x, y):

    params = ParameterSetup()
    classLabels = params.sampleClassLabels
    ratios = np.array(params.subsampleRatios)

    do_supersample = np.array(params.supersample)
    do_subsample = 1
    for ratio in ratios:
        if ratio == -1:
            do_subsample = 0
            break

    if do_supersample or do_subsample:
        minimumRatio = min(ratios)
        ratios = ratios / minimumRatio
        print('sampling ratios = ' + str(ratios))
        featureDim = x.shape[1]

        sorted_y = sorted(y)
        grouped = [[key, len(list(g))] for key, g in groupby(sorted_y)]
        for (key, elem_num) in grouped:
            print('  ' + key + ':' + str(elem_num))

        if do_supersample:
            max_class, max_elem_num = max(grouped)
            print('max_elem_num = ' + str(max_elem_num) + ' for max_class = ' + max_class)
            target_nums = [np.int(x) for x in np.floor(max_elem_num * ratios)]
        else:
            min_class, min_elem_num = max(grouped)
            print('min_elem_num = ' + str(min_elem_num) + ' for min_class = ' + min_class)
            target_nums = [np.int(x) for x in np.floor(min_elem_num * ratios)]

        print('target_nums = ' + str(target_nums))
        classID = 0
        sampled_x = np.zeros((0,featureDim), dtype=float)
        sampled_y = np.zeros((0))
        for targetClass in classLabels:
            print('sampling for classID = ' + str(classID) + ', class = ' + targetClass)
            isTarget = judgeIfTarget(y, targetClass)
            orderedIndices = np.arange(isTarget.shape[0])
            targetIDs = orderedIndices[isTarget==True]
            print('  target_nums[' + str(classID) + '] = ' + str(target_nums[classID]))

            sampledIDs = targetIDs
            print('    sampledIDs.shape = ' + str(sampledIDs.shape))

            if do_supersample:
                while sampledIDs.shape[0] <= target_nums[classID]:
                    orderedIndices = np.arange(isTarget.shape[0])
                    targetIDs = orderedIndices[isTarget==True]
                    sampledIDs = np.r_[sampledIDs, targetIDs]
                    print('    sampledIDs.shape = ' + str(sampledIDs.shape))

            sampledIDs = sampledIDs[:target_nums[classID]]
            print('    finally, sampledIDs.shape is reduced to: ' + str(sampledIDs.shape))

            print('  x[sampledIDs].shape = ' + str(x[sampledIDs].shape))
            # print('x[sampled_IDs] = ' + str(x[sampled_IDs]))
            sampled_x = np.r_[sampled_x, x[sampledIDs]]
            sampled_y = np.r_[sampled_y, y[sampledIDs]]
            # print('sampled_y = ' + str(sampled_y) + ', y[sampled_IDs] = ' + str(y[sampled_IDs]))
            # if sampled_y.shape[0] > 0:
            #    sampled_y = np.r_[sampled_y, y[sampled_IDs]]
            # else:
            #     sampled_y = y[sampled_IDs]
            # sampled_isTarget = isTarget[sampled_IDs]
            print('sampled_x.shape = ' + str(sampled_x.shape))
            print('sampled_y.shape = ' + str(sampled_y.shape))
            classID = classID + 1
        return (sampled_x, sampled_y)
    else:
        print('no supersampling nor subsampling')
        return (x, y)

'''
def subsample(x, y):

    params = ParameterSetup()
    classLabels = params.sampleClassLabels
    ratios = np.array(params.subsampleRatios)

    dosubsampling = 1
    for ratio in ratios:
        if ratio == -1:
            dosubsampling = 0
            break

    if dosubsampling:

        minimumRatio = min(ratios)
        ratios = ratios / minimumRatio
        print('subsampling ratios = ' + str(ratios))
        featureDim = x.shape[1]

        sorted_y = sorted(y)
        grouped = [[key, len(list(g))] for key, g in groupby(sorted_y)]
        min_class, min_elem_num = min(grouped)
        print('min_elem_num = ' + str(min_elem_num))
        target_nums = [np.int(x) for x in np.floor(min_elem_num * ratios)]
        print('target_nums = ' + str(target_nums))
        classID = 0
        subsampled_x = np.zeros((0,featureDim), dtype=float)
        subsampled_y = np.zeros((0))
        for targetClass in classLabels:
            print('subsampling for classID = ' + str(classID) + ', class = ' + targetClass)
            isTarget = judgeIfTarget(y, targetClass)
            orderedIndices = np.arange(isTarget.shape[0])
            targetIDs = orderedIndices[isTarget==True]
            print('target_nums[' + str(classID) + '] = ' + str(target_nums[classID]))
            subsampledIDs = targetIDs[:target_nums[classID]]
            print('x[subsampledIDs].shape = ' + str(x[subsampledIDs].shape))
            # print('x[subsampled_IDs] = ' + str(x[subsampled_IDs]))
            subsampled_x = np.r_[subsampled_x, x[subsampledIDs]]
            subsampled_y = np.r_[subsampled_y, y[subsampledIDs]]
            # print('subsampled_y = ' + str(subsampled_y) + ', y[subsampled_IDs] = ' + str(y[subsampled_IDs]))
            # if subsampled_y.shape[0] > 0:
            #    subsampled_y = np.r_[subsampled_y, y[subsampled_IDs]]
            # else:
            #     subsampled_y = y[subsampled_IDs]
            # subsampled_isTarget = isTarget[subsampled_IDs]
            print('subsampled_x.shape = ' + str(subsampled_x.shape))
            print('subsampled_y.shape = ' + str(subsampled_y.shape))
            classID = classID + 1
        return (subsampled_x, subsampled_y)
    else:
        print('no subsampling')
        return (x, y)
'''

def judgeIfTarget(y, targetClass):
    sampleNum = len(y)
    binary = np.zeros((sampleNum), dtype=np.bool)
    for i in range(sampleNum):
        if y[i] == targetClass:
            binary[i] = True
    return binary
