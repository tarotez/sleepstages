import numpy as np

def trimFeatures(params, dataMat):
  # print('params.useRawData = ', params.useRawData)
    if params.useRawData:
        if params.useFreqHisto and not params.useTime:
          dataMat = dataMat[:,:-1]
        elif not params.useFreqHisto and params.useTime:
          dataMat = np.c_[dataMat[:,:-params.additionalFeatureDim], dataMat[:,-1]]
        elif not params.useFreqHisto and not params.useTime:
          dataMat = dataMat[:,:-params.additionalFeatureDim]
    else:
        if params.useFreqHisto and not params.useTime:
          dataMat = dataMat[:,params.downsample_outputDim:-1]
        elif not params.useFreqHisto and params.useTime:
          dataMat = np.c_[dataMat[:,params.downsample_outputDim:-params.additionalFeatureDim], dataMat[:,-1]]
        elif not params.useFreqHisto and not params.useTime:
          dataMat = dataMat[:,params.downsample_outputDim:-params.additionalFeatureDim]
        else:
          dataMat = dataMat[:,params.downsample_outputDim:]

    return dataMat
