import numpy as np
import learningAlgorithm

class BalancedBagging(object):

    def __init__(self, dataClassOne, dataClassTwo, num_of_baseClf, L_rebalance_attributes):
        self.dataClassOne = dataClassOne
        self.dataClassTwo = dataClassTwo
        self.num_of_baseClf = num_of_baseClf
        self.L_rebalance_attributes = L_rebalance_attributes

    def getLearningAlogithmRebalance(self,L_rebalance_attributes):

        learningAlgorithmRebalance = learningAlgorithm.LearningAlgorithm(optimizer=L_rebalance_attributes['optimizer'],
                                                                     loss=L_rebalance_attributes['loss'],
                                                                     batch_size=L_rebalance_attributes['batch_size'],
                                                                     epoch=L_rebalance_attributes['epoch'],
                                                                     model_mode=L_rebalance_attributes['model_mode'],
                                                                         inputDim=L_rebalance_attributes['inputDim'])
        return learningAlgorithmRebalance




    # draw targetNum samples from each class randomly with replacement
    def resampleWithReplacement(self,dataPerClass, targetNum):
        X, y = dataPerClass
        dataIndex = X.index.tolist()
        resampledIndex = np.random.choice(dataIndex, size=targetNum, replace=True)
        new_X = X.loc[resampledIndex]
        new_y = y.loc[resampledIndex]
        return new_X, new_y

    # re-balance the two class data and train a list of base classifiers
    def balancedBagging(self):
        base_clf_list = []

        trainingDataSize = len(self.dataClassOne[0] + self.dataClassTwo[0])
        halfTrainingSize = int(trainingDataSize / 2)
        for i in range(self.num_of_baseClf):
            newXClassOne, newYClassOne = self.resampleWithReplacement(dataPerClass=self.dataClassOne,
                                                                 targetNum=halfTrainingSize)
            newXClassTwo, newYClassTwo = self.resampleWithReplacement(dataPerClass=self.dataClassTwo,
                                                                 targetNum=halfTrainingSize)
            rebalancedDataX = newXClassOne.append(newXClassTwo)
            rebalancedDataY = newYClassOne.append(newYClassTwo)
            #print('rebalanced data')
            #print(rebalancedDataX)
            #print(rebalancedDataY)
            curr_L_rebalance = self.getLearningAlogithmRebalance(self.L_rebalance_attributes)
            curr_L_rebalance.fit(rebalancedDataX, rebalancedDataY)
            base_clf_list.append(curr_L_rebalance)
        return base_clf_list
