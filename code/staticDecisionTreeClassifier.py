"""
This module builds a classifier that uses a static decision tree
where thresholds are trained but the order of using the attribues
are static.
"""

# Authors: Taro Tezuka <tez@sings.jp>

from __future__ import division

import warnings
# from abc import ABCMeta
# from abc import abstractmethod

import numpy as np

from sklearn.base import is_classifier
#from sklearn.externals import six
from sklearn.tree import DecisionTreeClassifier

from sampler import judgeIfTarget

class StaticDecisionTreeClassifier():

    def __init__(self):
        #-----------
        # for classical decision tree
        self.featureLabels = [['CV%'], ['Integral', 'FFT-DeltaRatio'], ['FFT-DeltaRatio', 'FFT-ThetaRatio'], ['FFT-DeltaPower', 'CV%'], ['FFT-ALphaPower', 'FFT-ThetaPower'], ['FFT-ThetaPower', 'Integral']]
        ### self.featureLabels = [['CV%', 'null'], ['Integral', 'FFT-DeltaRatio'], ['FFT-DeltaRatio', 'Integral'], ['FFT-DeltaPower', 'Integral'], ['FFT-ALphaPower', 'FFT-ThetaPower'], ['FFT-ThetaPower', 'Integral']]
        # self.featureLabels = [['CV%'], ['Integral', 'FFT-DeltaRatio'], ['FFT-DeltaRatio', 'FFT-ThetaPower', 'Integral'], ['FFT-DeltaPower', 'CV%'], ['FFT-ALphaPower', 'FFT-ThetaPower'], ['FFT-ThetaPower', 'Integral']]
        ###### self.featureLabels = [['CV%'], ['Integral', 'FFT-DeltaRatio'], ['FFT-DeltaRatio', 'CV%', 'Integral'], ['FFT-DeltaPower', 'Integral'], ['FFT-ALphaPower', 'FFT-ThetaPower'], ['FFT-ThetaPower', 'Integral']]
        ### self.featureLabels = ['CV%', 'null', 'Integral', 'FFT-DeltaRatio', 'FFT-DeltaRatio', 'CV%', 'FFT-DeltaPower', 'Integral', 'FFT-ALphaPower', 'FFT-ThetaPower', 'FFT-ThetaPower', 'Integral']
        ### self.featureLabels = ['CV%', 'null', 'Integral', 'FFT-DeltaRatio', 'FFT-DeltaRatio', 'FFT-ThetaPower', 'FFT-DeltaPower', 'Integral', 'FFT-ALphaPower', 'FFT-ThetaPower', 'FFT-ThetaPower', 'Integral']
        # downsampleRatio sets the ratio LargerClassSampleNum / SmallerClassSampleNum.
        # if set to -1, no downsampling will be conducted.
        # self.downsampleRatio = [1, 1, 1, 1, 1, 1]
        self.downsampleRatio = [-1, -1, -1, -1, -1, -1]
        self.targetClassSeq = ['M', 'S', 'R', 'S', 'W', 'W']
        self.ruleOnOffs = [[1],[1,1],[1,1],[1,1],[1,1],[1,1]]

    def extract(self, old_list, binary):
        sampleNum = len(old_list)
        new_list = []
        for i in range(sampleNum):
            if binary[i]:
                new_list.append(old_list[i])
        return new_list

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None, staticParamsBinary=[], staticParamsVals=[]):
        """Build a decision tree classifier from the training set (X, y).
        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels) as integers or strings.
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        X_idx_sorted : array-like, shape = [n_samples, n_features], optional
            The indexes of the sorted training input samples. If many tree
            are grown on the same dataset, this allows the ordering to be
            cached between trees. If None, the data will be sorted here.
            Don't use this parameter unless you know what to do.
        Returns
        -------
        self : object
            Returns self.
        """

        layerNum = len(self.ruleOnOffs)

        remainingX = X
        remainingy = y
        self.oneLayerTrees = []
        print(' ')
        treeID = 0
        for layer in range(layerNum):
            print(' ')
            print('layer = ' + str(layer))
            # print('len(remainingX) = ' + str(len(remainingX)))
            # print('remainingX[:4,:4] = ' + str(remainingX[:4,:4]))
            print('len(remainingy) = ' + str(len(remainingy)))
            print('remainingy[:8] = ' + str(remainingy[:8]))
            targetClass = self.targetClassSeq[layer]
            isTarget = judgeIfTarget(remainingy, targetClass)
            print('targetClass = ' + targetClass + ', samples are ' + str(np.sum(isTarget)))
            condNum = len(self.ruleOnOffs[layer])
            print('   condNum = ' + str(condNum))
            predsList = []
            for condID in range(condNum):
                ### treeID = (layer*2)+condID
                print('treeID = ' + str(treeID))
                oneLayerTree = DecisionTreeClassifier(max_depth=1)
                print('remainingX.shape = ' + str(remainingX.shape))
                # print('isTarget.shape = ' + str(isTarget.shape))
                samples = remainingX[:,treeID].reshape(-1,1)

                #----
                # downsample to make the number of samples even for classes
                if self.downsampleRatio[layer] >= 0:
                    print('downsampling for treeID = ' + str(treeID))
                    trueNum = np.sum(isTarget == True)
                    falseNum = isTarget.shape[0] - trueNum
                    print('trueNum = ' + str(trueNum) + ', falseNum = ' + str(falseNum))
                    orderedIndices = np.arange(isTarget.shape[0])
                    downsampled_IDs = orderedIndices
                    if trueNum > falseNum and falseNum > 0:
                        randomizedTrues = np.random.permutation(orderedIndices[isTarget==True])
                        downsampledNum = falseNum * self.downsampleRatio[layer]
                        downsampled_IDs = np.r_[randomizedTrues[:downsampledNum], orderedIndices[isTarget==False]]
                    elif trueNum < falseNum and trueNum > 0:
                        randomizedFalses = np.random.permutation(orderedIndices[isTarget==False])
                        downsampledNum = trueNum * self.downsampleRatio[layer]
                        downsampled_IDs = np.r_[randomizedFalses[:downsampledNum], orderedIndices[isTarget==True]]
                    downsampled_samples = samples[downsampled_IDs]
                    downsampled_isTarget = isTarget[downsampled_IDs]
                else:
                    downsampled_samples = samples
                    downsampled_isTarget = isTarget
                print('downsampled_samples.shape = ' + str(downsampled_samples.shape))
                print('downsampled_isTarget.shape = ' + str(downsampled_isTarget.shape))
                #---
                # fit the tree
                oneLayerTree.fit(downsampled_samples, downsampled_isTarget)

                #----
                # show learned thresholds
                features = oneLayerTree.tree_.feature
                threshs = oneLayerTree.tree_.threshold
                for node in range(1):
                    print('condID = ' + str(condID) + ', node = ' + str(node) + ', feature = ' + str(features[node]) + ', thresh = ' + str(threshs[node]))

                #----
                # show predictions using each condition
                preds = oneLayerTree.predict(samples)
                predsList.append(preds)
                matched = isTarget == preds
                t2t = (isTarget == True) & (preds == True)
                t2o = (isTarget == True) & (preds == False)
                o2t = (isTarget == False) & (preds == True)
                o2o = (isTarget == False) & (preds == False)
                # print('correct = ' + str(isTarget[:15]))
                # print('predict = ' + str(preds[:15]))
                # print('correct among assigned = ' + str(np.sum(trueIdentified) / np.sum(preds)))
                print('Target: ' + str(np.sum(isTarget)) + ', Others:' + str(np.sum(~isTarget)))
                print('Target -> Target: ' + str(np.sum(t2t)) + ', Target -> Others: ' + str(np.sum(t2o)))
                print('Others -> Target: ' + str(np.sum(o2t)) + ', Others -> Others: ' + str(np.sum(o2o)))
                print('ratio of t2t / (t2t + t2o) = ' + str(np.sum(t2t) / np.sum(isTarget)))
                print('ratio of o2o / (o2t + o2o) = ' + str(np.sum(o2o) / np.sum(~isTarget)))
                print('ratioMatched = ' + str(np.sum(matched) / matched.shape[0]))

                self.oneLayerTrees.append(oneLayerTree)
                # print('isTarget = ' + str(isTarget))
                # print('np.sum(isTarget) = ' + str(np.sum(isTarget)))
                treeID = treeID + 1

            if condNum == 2:
                predsProduct = self.getPredsProduct(layer, predsList[0], predsList[1])
            else:
                predsProduct = self.vote(layer, predsList)

            matched = isTarget == predsProduct
            t2t = (isTarget == True) & (predsProduct == True)
            o2o = (isTarget == False) & (predsProduct == False)
            # print('correct = ' + str(isTarget[:15]))
            # print('predict = ' + str(preds[:15]))
            print('ratio of t2t / (t2t + t2o) = ' + str(np.sum(t2t) / np.sum(isTarget)))
            print('ratio of o2o / (o2t + o2o) = ' + str(np.sum(o2o) / np.sum(~isTarget)))
            print('ratioMatched = ' + str(np.sum(matched) / matched.shape[0]))

            remainingX = remainingX[~predsProduct,:]
            remainingy = self.extract(remainingy,~predsProduct)
            # remainingX = remainingX[~isTarget,:]
            # remainingy = self.extract(remainingy,~isTarget)
            print('remainingy[:100] = ' + str(remainingy[:50]))
            if len(remainingy) == 0:
                break

    def vote(self, layer, predsList):
        # print('layer = ' + str(layer))
        elemNum = predsList[0].shape[0]
        voteResult = np.zeros(elemNum, dtype=bool)
        for i in range(elemNum):
            activeRuleNum = 0
            ruleID = 0
            trueCounter = 0
            for preds in predsList:
                # print('  layer = ' + str(layer) + ', activeRuleNum = ' + str(activeRuleNum))
                if self.ruleOnOffs[layer][ruleID]:
                    activeRuleNum += 1
                    if preds[i] == True:
                        trueCounter += 1
                        # print('  layer = ' + str(layer) + ', trueCounter = ' + str(trueCounter))
                ruleID += 1
            if trueCounter >= activeRuleNum / 2:
                voteResult[i] = True
        return voteResult

    def getPredsProduct(self, layer, preds0, preds1):
        # show predictions using two conditions
        if self.ruleOnOffs[layer][0]:
            if self.ruleOnOffs[layer][1]:
                predsProduct = np.multiply(preds0, preds1)
            else:
                predsProduct = preds0
        else:
            if self.ruleOnOffs[layer][1]:
                predsProduct = preds1
            else:
                predsProduct = np.zeros(preds0.shape[0], dtype=bool)
        return predsProduct

    def predict(self, X, check_input=True):
        """Predict class or regression value for X.
        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """

        # print('X.shape = ' + str(X.shape))
        # print('oneLayerTrees.shape = ' + str(len(self.oneLayerTrees)))
        sampleNum = X.shape[0]
        predList = []
        # predForOneAttr = [[],[]]
        for sampleID in range(sampleNum):
            notClassified = True
            # print(' ')
            layerNum = len(self.ruleOnOffs)
            treeID = 0
            for layer in range(layerNum):
                # print(' ')
                # print('in predict():')
                # print('sampleID = ' + str(sampleID) + ', layer = ' + str(layer))
                # print('X.shape = ' + str(X.shape))
                # print('X = ' + str(X))
                condNum = len(self.ruleOnOffs[layer])
                #############
                # print('ruleOnOffs = ' + str(self.ruleOnOffs))
                # print('**** layer = ' + str(layer) + ', condNum = ' + str(condNum))
                predsList = []
                for condID in range(condNum):
                    # treeID = (layer*2)+condID
                    # print('layer = ' + str(layer) + ', treeID = ' + str(treeID))
                    preds = self.oneLayerTrees[treeID].predict(np.array([[X[sampleID,treeID]]]))
                    predsList.append(preds)
                    # print('in layer = ' + str(layer) + ', treeID = ' + str(treeID) + ', pred' + str(condID) + ' = ' + str(preds))
                    treeID = treeID + 1

                ### below should be element-wise product
                ### also, pred must be binary
                # print('layer ' + str(layer) + ': (' + str(predForOneAttr[0]) + ", " + str(predForOneAttr[1]) + ")")
                if condNum == 2:
                    predsProduct = self.getPredsProduct(layer, predsList[0], predsList[1])
                else:
                    predsProduct = self.vote(layer, predsList)
                # print('in layer = ' + str(layer) + ', predsProduct = ' + str(predsProduct))

                if predsProduct[0]:
                    '''
                    print('predsProduct[0] == True in layer = ' + str(layer))
                    for condID in range(2):
                        thresh = self.getThresh(layer, condID)
                        treeID = (layer*2)+condID
                        x = X[sampleID,treeID]
                        print('  thresh' + str(condID) + ' = ' + str(thresh) + ', x = ' + str(x))
                        if thresh > x:
                            print('x is smaller!')
                        else:
                            print('x is larger!')
                    '''
                    predList.append(self.targetClassSeq[layer])
                    notClassified = False
                    break

            if notClassified:
                # print('predList.append(P)')
                predList.append('P')

        return predList

    '''
    def setThresh(self, layer, condID, thresh):
        treeID = (layer*2) + condID
        self.oneLayerTrees[treeID].tree_.threshold = [thresh]

    def getThresh(self, layer, condID):
        treeID = (layer*2) + condID
        features = self.oneLayerTrees[treeID].tree_.feature
        threshs = self.oneLayerTrees[treeID].tree_.threshold
        return threshs[0]
        '''
    '''
    def showThresh(self):
        """ Print outs parameters
        """
        layerNum = np.int(np.floor(len(self.oneLayerTrees)/2))
        for layer in range(layerNum):
            for condID in range(2):
                treeID = (layer*2) + condID
                #vprint('layer: ' + str(layer) + ', condID = ' + str(condID) + ', treeID = ' + str(treeID))
                features = self.oneLayerTrees[treeID].tree_.feature
                threshs = self.oneLayerTrees[treeID].tree_.threshold
                for node in range(1):
                    print('L-' + str(layer) + ', targ: ' + self.targetClassSeq[layer] + ', attr: ' + self.ruleOnOffs[layer][condID] + ', thresh: ' + str(threshs[node]))
        print('ruleOnOffs = ' + str(self.ruleOnOffs))
        print('downsampleRatio = ' + str(self.downsampleRatio))
    '''
