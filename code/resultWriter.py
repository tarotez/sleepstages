import numpy as np

class ResultWriter():

    def __init__():
        pass

    def write():


        #-----
        # show the summary (average) of the result
        print('classifier type = ' + str(classifierType))
        print('classifier parameter = ' + str(classifierParams[paramID]))
        print('pastStageLookUpNum = ' + str(params.pastStageLookUpNum))
        print('useEMG = ' + str(params.useEMG))
        print('emgTimeFrameNum = ' + str(params.emgTimeFrameNum))
        print('binWidth4freqHisto = ' + str(params.binWidth4freqHisto))
        sensitivityMean = np.zeros((labelNum), dtype=float)
        specificityMean = np.zeros((labelNum), dtype=float)
        accuracyMean = np.zeros((labelNum), dtype=float)
        for labelID in range(labelNum):
            sensitivityMean[labelID] = sensitivitySum[labelID] / fileNum
            specificityMean[labelID] = specificitySum[labelID] / fileNum
            accuracyMean[labelID] = accuracySum[labelID] / fileNum
            print('  stageLabel = ' + stageLabels[labelID] + ', sensitivity = ' + "{0:.3f}".format(sensitivityMean[labelID]) + ', specificity = ' + "{0:.3f}".format(specificityMean[labelID]) + ', accuracy = ' + "{0:.3f}".format(accuracyMean[labelID]))
        precisionMean = precisionSum / fileNum
        print('  precision = ' + "{0:.5f}".format(precisionMean))


