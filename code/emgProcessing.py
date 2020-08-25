from __future__ import print_function
import numpy as np
from numpy import ndarray

def emg2feature(emg, emgTimeFrameNum):
    emgFeature = np.zeros((emgTimeFrameNum,1), dtype=float)
    emgLen = emg.shape[0]
    emgShortSegmentLen = np.int(np.floor(emgLen / emgTimeFrameNum))
    for i in range(emgTimeFrameNum):
        startTime = i * emgShortSegmentLen
        endTime = (i + 1) * emgShortSegmentLen
        emgShortSegment = emg[startTime:endTime]
        emgFeature[i,0] = np.mean(np.abs(emgShortSegment))
    return emgFeature
