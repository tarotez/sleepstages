import sys
from parameterSetup import ParameterSetup

if __name__ == '__main__':
    args = sys.argv
    classifierIDs = args[1:]
    res = []
    for classifierID in classifierIDs:
        paramFileName = 'params.' + classifierID + '.json'
        params = ParameterSetup(paramFileName=paramFileName)
        stageLabels = params.stageLabels4evaluation
        print('stageLabels =', stageLabels)
        res.append(classifierID + ', ' + str(params.useRawData) + ', ' + str(params.useFreqHisto) + ', ' + str(params.useTime))
    for line in res:
        print(line)
