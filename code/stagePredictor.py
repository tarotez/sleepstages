from __future__ import print_function
import pickle
import warnings
import numpy as np
from emgProcessing import emg2feature
from stageLabelAndOneHot import constructPastStagesOneHots, oneHot2stageLabel
# from processFeatures import trimFeatures

class StagePredictor(object):

    def __init__(self, params, extractor, classifier, classifierDir, classifierID, markovOrderForPrediction):

        # parameters for signal processing
        self.params = params
        windowSizeInSec = params.windowSizeInSec   # size of window in time for estimating the state
        samplingFreq = params.samplingFreq   # sampling frequency of data

        # parameters for using history
        self.emgTimeFrameNum = params.emgTimeFrameNum
        self.preContextSize = params.preContextSize
        self.pastStageLookUpNum = params.pastStageLookUpNum
        ### self.stageNum = len(params.stageLabel2stageID)
        self.stageNum = params.maximumStageNum

        # dictionary for label correction
        self.labelCorrectionDict = params.labelCorrectionDict

        # self.classifier.showThresh()
        samplePointNum = samplingFreq * windowSizeInSec   # window size. data is sampled at 128 Hz, so 1280 sample points = 10 sec.
        self.time_step = 1 / samplingFreq
        # print('samplePointNum = ' + str(samplePointNum))
        # self.past_eeg = np.empty((samplePointNum, 0), dtype = np.float)
        # self.featureDim = self.binNum4spectrum + self.useEMG + self.pastStageLookUpNum   # [freqHisto, emgFeature, pastStage]
        # self.past_features = np.empty((self.featureDim, 0), dtype = np.float)
        # print('in __init__, self.past_eeg.shape = ' + str(self.past_eeg.shape))
        self.stageCnt = 0
        self.pastStages_L = []
        self.pastFeatures_L = []
        self.useEMG = params.useEMG
        self.markovOrder = markovOrderForPrediction
        self.extractor = extractor
        self.classifier = classifier

        if params.useEMG:
            label4EMG = params.label4withEMG
        else:
            label4EMG = params.label4withoutEMG

        # self.excludedFileID = excludedFileID
        # if self.excludedFileID != '':
        #    featurePath = open(params.classifierDir + '/' + params.featureFilePrefix + '.' + params.extractorType + '.' + label4EMG + '.' + self.excludedFileID + '.pkl', 'rb')
        #    self.featureVecAll = pickle.load(featurePath)
        #    # print('self.featureVecAll.shape = ' + str(self.featureVecAll.shape))

        # get transition probability matrix from transition matrix (counts)
        # print('in stagePredictor, self.markovOrder =', self.markovOrder)
        if self.markovOrder > 0:

            if len(classifierID) > 0:
                transitionTensorPath = open(classifierDir + '/transitionTensor.' + classifierID + '.pkl', 'rb')
            else:
                transitionTensorPath = open(classifierDir + '/transitionTensor.pkl', 'rb')

            transitionTensor = pickle.load(transitionTensorPath)
            # print('% transitionTensor.shape = ' + str(transitionTensor.shape))

            # project transitionTensor to the Markov order specified by self.markovOrder
            # print('# before projection, transitionTensor.shape = ', transitionTensor.shape)
            for i in range(len(transitionTensor.shape) - self.markovOrder - 1):
                transitionTensor = np.sum(transitionTensor,axis=-1)
            # print('# after projection, transitionTensor.shape = ', transitionTensor.shape)

            # remove states that occurs less than 1%.
            # necessary to avoid division by a small number
            thresh = 0.00001 * np.mean(transitionTensor)
            transitionTensor[transitionTensor < thresh] = 0

            sumTensor = transitionTensor
            for i in range(self.markovOrder):
                # print('sum axis = ' + str(i))
                sumTensor = np.sum(sumTensor, axis=-1)
            transitionPresentTimeSumVec = sumTensor
            # print('transitionPresentTimeSumVec.shape = ' + str(transitionPresentTimeSumVec.shape))

            '''
            # prints transition matrix
            print('transitionTensor:')
            for mat in transitionTensor:
                for vec in mat:
                    print(' ' + str(vec))
                    print(' ')
            print('')
            '''

            # compute present distribution. must be checked
            self.presentDistribution = transitionPresentTimeSumVec / np.sum(transitionPresentTimeSumVec)
            self.pastDistributionL = []

            # print('transitionTensor.shape', transitionTensor.shape)
            # print('np.sum(transitionTensor)', np.sum(transitionTensor))
            sumTensor = np.sum(transitionTensor, axis=0)
            # print('sumTensor.shape', sumTensor.shape)
            # print('np.sum(sumTensor)', np.sum(sumTensor))
            # print('sumTensor', sumTensor)

            freqVector = transitionTensor
            for i in range(self.markovOrder):
                freqVector = np.sum(freqVector,axis=-1)

            # print('# freqVector = ', freqVector)
            zeroth_order_markov_distrib = freqVector / np.sum(freqVector)
            # print('zeroth_order_markov_distrib = ', zeroth_order_markov_distrib)
            # print('zeroth_order_markov_distrib.sum() = ', zeroth_order_markov_distrib.sum())

            for i in range(self.markovOrder):
                pastDistribution = self.presentDistribution
                self.pastDistributionL.append(pastDistribution)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.transitionProbTensor = np.divide(transitionTensor, np.sum(transitionTensor, axis=0))
            self.transitionProbTensor[np.isnan(self.transitionProbTensor)] = 0
            sumTensor = np.sum(self.transitionProbTensor,axis=0)
            # print('self.transitionProbTensor.shape', self.transitionProbTensor.shape)
            # print('sumTensor', sumTensor)

            # replace distributions with all zero probability with 0th-order Markov distributions,
            # that is, replace p(a|b,c,d) by p(a) if sequence (b,c,d) does not appear.
            labelNum = sumTensor.shape[0]
            newTransitionProbTensorL = []
            reshapedTransitionProbTensor = self.transitionProbTensor.reshape((labelNum,-1)).transpose()
            reshapedSumTensor = sumTensor.reshape((-1))
            # print('****** reshapedTransitionProbTensor.shape = ', reshapedTransitionProbTensor.shape)
            for transitionProbVec, sum in zip(reshapedTransitionProbTensor, reshapedSumTensor):
                # print('# sum = ', sum)
                # print('# transitionProbVec.shape = ', transitionProbVec.shape)
                # print('# zeroth_order_markov_distrib = ', zeroth_order_markov_distrib)
                if sum == 0:
                    # print('# in sum == 0, transitionProbVec = ', transitionProbVec)
                    newTransitionProbTensorL.append(zeroth_order_markov_distrib)
                else:
                    newTransitionProbTensorL.append(transitionProbVec)

            flatNewTransitionProbTensor = np.array(newTransitionProbTensorL).reshape(reshapedTransitionProbTensor.shape)
            # print('flatNewTransitionProbTensor.shape = ', flatNewTransitionProbTensor.shape)
            # print('flatNewTransitionProbTensor = ', flatNewTransitionProbTensor)
            newTransitionProbTensor = flatNewTransitionProbTensor.transpose().reshape(self.transitionProbTensor.shape)
            # print('newTransitionProbTensor.shape = ', newTransitionProbTensor)
            diffTensor = newTransitionProbTensor - self.transitionProbTensor
            # print('$$$$$$ diffTensor = ', diffTensor)
            # print('$$$$$$ zeroth_order_markov_distrib = ', zeroth_order_markov_distrib)
            self.transitionProbTensor = newTransitionProbTensor
            # newSumTensor = np.sum(self.transitionProbTensor,axis=0)
            # print('#% newSumTensor (all of its elements should equal to 1) = ', newSumTensor)

            ''''
            # prints transition prob matrix
            print('transitionProbTensor:')
            for mat in self.transitionProbTensor:
                for vec in mat:
                    print(' ' + str(vec))
                    print(' ')
            print('')
            print('Checking transitionProbTensor. The following numbers should all be one.')
            sumOfProbs = np.sum(self.transitionProbTensor, axis=0)
            if np.sum(np.abs(sumOfProbs - np.ones(sumOfProbs.shape))) > 0.0001:
                if np.sum(sumOfProbs) > 0.00001:
                    for i in range(1):
                        print('ERROR. probs did not add up to 1. It was ' + str(sumOfProbs))
            #----
            # sample initial state
            sampled = np.random.uniform()
            accum = 0
            for stateID in range(len(self.presentDistribution)):
                accum += self.presentDistribution[stateID]
                if accum > sampled:
                    self.pastStateID = stateID
                    break
            print('initial pastStateID = ' + str(self.pastStateID))
            '''

    def batch_predict(self, records, timeStampSegments, stageLabels4evaluation, stageLabel2stageID):

        # print('records.shape = ' + str(records.shape))
        # print('timeStampSegments.shape = ' + str(timeStampSegments.shape))

        # if self.excludedFileID != '':
        #    featureMat = self.featureVecAll
        #    print('featureVec constructed for ' + self.excludedFileID + '.')
        # else:
        # featureMat_L = []
        wID = 0
        for one_record in records:
            if self.useEMG == 0:
                if len(records.shape) == 3:
                    eegSegment = one_record[:,0]
                else:
                    eegSegment = one_record
            else:
                if len(records.shape) == 2:
                    self.useEMG = 0
                    eegSegment = one_record
                else:
                    self.useEMG = 1
                    eegSegment = one_record[:,0]
            # print('in stagePredictor, eegSegment =', eegSetment)
            timeStampSegment = timeStampSegments[wID]
            featureIncrement = np.array([self.extractor.getFeatures(eegSegment, timeStampSegment, self.time_step)])
            # print('&&& before trimming, featureIncrement.shape =', featureIncrement.shape)
            # featureIncrement = trimFeatures(self.params, featureIncrement)
            # print('&&& after trimming, featureIncrement.shape =', featureIncrement.shape)

            if wID == 0:
                featureTensor = featureIncrement
            else:
                # featureMat_L.append(self.extractor.getFeatures(eegSegment, timeStampSegment, self.time_step))
                # print('featureTensor.shape = ' + str(featureTensor.shape) + ', featureIncrement.shape = ' + str(featureIncrement.shape))
                featureTensor = np.r_[featureTensor, featureIncrement]
            wID += 1
            # featureMat = np.array(featureMat_L)

        # print('featureTensor.shape = ' + str(featureTensor.shape))
        # print('featureTensor = ' + str(featureTensor))
        # featureTensorTransposed = featureTensor.transpose([0,2,1])
        # print('featureTensorTransposed.shape = ' + str(featureTensorTransposed.shape))
        # print('before classifier.predict in stagePredictor')
        ### y_pred_origs = self.classifier.predict(featureTensorTransposed)
        y_pred_origs = self.classifier.predict(featureTensor)
        # print('after classifier.predict in StagePredictor')
        # print('np.array(y_pred_origs).shape = ' + str(np.array(y_pred_origs).shape))

        # y_pred_L = [self.labelCorrectionDict[y_pred_orig[0]] for y_pred_orig in y_pred_origs]
        # [print('y_pred_orig = ' + str(y_pred_orig)) for y_pred_orig in y_pred_origs]
        y_pred_L = [self.labelCorrectionDict[oneHot2stageLabel(y_pred_orig, stageLabels4evaluation, stageLabel2stageID)] for y_pred_orig in y_pred_origs]
        # print('y_pred_L = ' + str(y_pred_L))
        return y_pred_L

    def predict(self, one_record, timeStampSegment, stageLabels4evaluation, stageLabel2stageID, wID=-1):

        # print('one_record = ' + str(one_record))
        # print('one_record.shape = ' + str(one_record.shape))
        if self.useEMG == 0:
            if len(one_record.shape) == 2:
                ##### eegSegment = one_record[:]
                eegSegment = one_record[:,0]
            else:
                eegSegment = one_record
        else:
            if len(one_record.shape) == 1:
                self.useEMG = 0
                eegSegment = one_record
            else:
                self.useEMG = 1
                eegSegment = one_record[:,0]
        # print('in stagePredictor, useEMG = ' + str(self.useEMG))
        # print('in stageP, eegSegment = ' + str(eegSegment))
        # timestamps = np.array(timestampsL)
        # self.past_eeg = np.c_[self.past_eeg, eeg]
        # print('timeStampSegmeng = ' + str(timeStampSegment))

        featureVec = self.extractor.getFeatures(eegSegment, timeStampSegment, self.time_step)
        # print('@@@ in stagePredictor, featureVec.shape = ' + str(featureVec.shape))
        # print('featureVec = ' + str(featureVec))
        featureMat = np.array([featureVec])
        # print('&&& before trimming, featureMat.shape =', featureMat.shape)
        # featureMat = trimFeatures(self.params, featureMat)
        # print('&&& after trimming, featureMat.shape =', featureMat.shape)

        if self.useEMG:
            emgSegment = one_record[:,1]
            emgFeature = emg2feature(emgSegment, self.emgTimeFrameNum)
        # print('emgFeature = ' + str(emgFeature))
        # print('pastStageLookUpNum = ' + str(self.pastStageLookUpNum))
        # print('stageNum = ' + str(self.stageNum))
        # print('product = ' + str(self.pastStageLookUpNum * self.stageNum))

        if self.stageCnt >= self.pastStageLookUpNum:
            pastStagesOneHots = constructPastStagesOneHots(self.pastStages_L, self.stageCnt, self.pastStageLookUpNum, self.stageNum)
        else:
            pastStagesOneHots = np.zeros((self.pastStageLookUpNum * self.stageNum, 1), dtype=np.float)

        ### pastStagesOneHots = np.zeros((self.pastStageLookUpNum * self.stageNum, 1), dtype=np.float)
        # print('original: pastStagesOneHots.shape = ' + str(pastStagesOneHots.shape))
        # print('stageCnt = ' + str(self.stageCnt))
        ### if self.stageCnt >= self.pastStageLookUpNum:
            ### for offset in range(1, self.pastStageLookUpNum + 1):
                ### pastStage = self.pastStages_L[self.stageCnt - offset]
                # print('offset = ' + str(offset) + ', pastStage = ' + str(pastStage))
                ### oneHot = stageLabel2oneHot(pastStage, maximumStageNum)
                # print('   offset = ' + str(offset))
                # print('   oneHot.shape = ' + str(oneHot.shape))
                # print('   pastStagesOneHots.shape = ' + str(pastStagesOneHots.shape))
                # print('   index = ' + str((offset - 1) * self.stageNum) + ':' + str(offset * self.stageNum))
        # else:
        #    oneHot = np.zeros((self.pastStageLookUpNum * self.stageNum, 1), dtype=np.float)

        # print('after loop: pastStagesOneHots.shape = ' + str(pastStagesOneHots.shape))
        # print('oneHot.shape = ' + str(oneHot.shape))

        if self.extractor.extractorType == 'classical' or self.extractor.extractorType.startswith('wavelet'):
            features_part = featureMat
        else:
            if self.useEMG:
                # print('featureMat.shape = ' + str(featureMat.shape))
                # print('emgFeature.shape = ' + str(emgFeature.shape))
                features_part = np.c_[featureMat, emgFeature.transpose()]
            else:
                features_part = featureMat

        # print('self.pastStageLookUpNum = ' + str(self.pastStageLookUpNum))
        if self.pastStageLookUpNum > 0:
            # print('features_part.shape = ' + str(features_part.shape))
            # print('pastStagesOneHots.shape = ' + str(pastStagesOneHots.shape))
            features = np.c_[features_part, np.transpose(pastStagesOneHots)]
        else:
            features = features_part
        # features = np.transpose(features)
        self.stageCnt = self.stageCnt + 1
        # print('features.shape =', features.shape)
        if self.params.classifierType == 'deep' and self.params.networkType == 'cnn_lstm':
            self.pastFeatures_L.append(features)
            # print('len(self.pastFeatures_L) =', len(self.pastFeatures_L))
            if len(self.pastFeatures_L) > self.params.torch_lstm_length:
                self.pastFeatures_L = self.pastFeatures_L[1:]
            if len(self.pastFeatures_L) >= self.params.torch_lstm_length:
                features_with_past = np.array(self.pastFeatures_L)
                # print('features_with_past.shape =', features_with_past.shape)
                y_pred_orig = self.classifier.predict(features_with_past)
            else:
                y_pred_orig = '?'
        else:
            #---------------
            # predict using the trained classifier
            # print('before classifier.predict in stagePredictor')
            # print('****** in stagePredictor.predict(), features.shape = ' + str(features.shape))
            y_pred_orig = self.classifier.predict(features)
            # print('after classifier.predict in StagePredictor')
            # print('y_pred_orig = ' + str(y_pred_orig))
            # y_pred = self.labelCorrectionDict[y_pred_orig[0]]
            # print('original y_pred_orig = ' + str(y_pred_orig))

        # print('y = ', y_pred_orig, '; ', sep='', end='')

        if self.markovOrder > 0:
            # prints transition matrix
            # print('productMatrix:')
            # for vec in productMatrix:
                # print("  " + str(vec))
            # print('productDistribution = ' + str(productDistribution))
            # print('')
            # print('self.presentDistribution = ' + str(self.presentDistribution))
            productDistribution = self.transitionProbTensor
            for i in range(self.markovOrder,0,-1):
                # print('productDistribution.shape = ' + str(productDistribution.shape))
                # print('productDistribution = ' + str(productDistribution))
                # print('# np.sum(productDistribution, axis=0) = ', np.sum(productDistribution, axis=0))
                # print('i = ' + str(i))
                productDistribution = np.sum(np.multiply(productDistribution, self.pastDistributionL[i-1]), axis=i)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coeffVec = np.divide(productDistribution, self.presentDistribution)
            coeffVec[np.isnan(coeffVec)] = 0
            # print('coeffVec = ' + str(coeffVec))
            y_pred_modified = np.multiply(y_pred_orig, coeffVec)
            # print('y_pred_modified = ' + str(y_pred_modified))

            # print('#%#%#% self.markovOrder = ', self.markovOrder)
            # print('#%#%#% before: len(self.pastDistributionL) =', len(self.pastDistributionL))
            # print('#%#%#% before: self.pastDistributionL =', self.pastDistributionL)
            for i in range(len(self.pastDistributionL)-1,0,-1):
                self.pastDistributionL[i] = self.pastDistributionL[i-1]
            if len(self.pastDistributionL) > 0:
                self.pastDistributionL[0] = self.presentDistribution
            # print('#%#%#% after: len(self.pastDistributionL) =', len(self.pastDistributionL))
            # print('#%#%#% after: self.pastDistributionL =', self.pastDistributionL)

        else:
            y_pred_modified = y_pred_orig

        # print('before labelCorrection, y_pred_modified =', y_pred_modified)
        def correct_label(items):
            return self.labelCorrectionDict[oneHot2stageLabel(items[0], stageLabels4evaluation, stageLabel2stageID)]

        if self.params.classifierType == 'deep':
            if type(y_pred_modified) != list and type(y_pred_modified) != np.ndarray:
                if y_pred_modified == '?':
                    y_pred = y_pred_modified
                else:
                    y_pred = correct_label(y_pred_modified)
            else:
                y_pred = correct_label(y_pred_modified)
            # print('after labelCorrection, y_pred =', y_pred)
            # print('y_pred = ' + str(y_pred))
        else:
            y_pred = self.params.labelCorrectionDict[y_pred_modified[0]]

        # print(y_pred, end='')
        self.pastStages_L.append(y_pred_modified[0])
        return y_pred
