import numpy as np
import ssm
import balanced_bagging

class NoiseFilter(object):

    def __init__(self, num_of_baseClf=5, threshold=0.5, Q=0.1, beta=50):
        self.num_of_baseClf = num_of_baseClf
        self.threshold = threshold
        self.Q = Q
        self.beta = beta

    # SSM-based Noise Filter, return the filtered dataset
    def noiseFilter(self,data,L_rebalance_attributes):
        st_sm = ssm.SSM(Q=self.Q, beta=self.beta)
        X, Y = data
        remove_indexList = []
        Y_List = list(set(Y[0]))
        for i in range(len(Y_List) - 1):
            for j in range(i + 1, len(Y_List)):
                print('do the balance bagging on pair: ('+str(Y_List[i])+'\t'+str(Y_List[j])+')')
                # re-balance the selected two class data and train a list of clf based on it
                classOneIndexList = Y[Y[0] == Y_List[i]].index.tolist()
                classTwoIndexList = Y[Y[0] == Y_List[j]].index.tolist()
                if len(classOneIndexList) == 0 or len(classTwoIndexList) == 0:
                    print('error, classIndexList is empty (inside the noiseFilter function)')
                    return
                classOneX = X.ix[classOneIndexList]
                classTwoX = X.ix[classTwoIndexList]
                classOneY = Y.ix[classOneIndexList]
                classTwoY = Y.ix[classTwoIndexList]
                classOneY = classOneY.replace(classOneY[0:1], 0)
                classTwoY = classTwoY.replace(classTwoY[0:1], 1)

                ba_ba = balanced_bagging.BalancedBagging((classOneX, classOneY), (classTwoX, classTwoY),
                                                         self.num_of_baseClf, L_rebalance_attributes)
                clf_list = ba_ba.balancedBagging()
                # print("check if clfs are the same")
                # for m in range(len(clf_list)):
                #     print("clf ",m)
                #     print(clf_list[m].get_weights())

                curr_removed_indexList = []
                # get the avg ssm for each training sample from class i and j
                # and record those data that should be removed (whose ssm>threshold)
                for index, row in classOneX.iterrows():
                    curr_x = np.array(row)
                    curr_y = np.array(classOneY.ix[index])
                    #print(st_sm.avg_ssm_eachX(curr_x, curr_y, clf_list))
                    if st_sm.avg_ssm_eachX(curr_x, curr_y, clf_list) > self.threshold:
                        curr_removed_indexList.append(index)
                for index, row in classTwoX.iterrows():
                    curr_x = np.array(row)
                    curr_y = np.array(classTwoY.ix[index])
                    #print(st_sm.avg_ssm_eachX(curr_x, curr_y, clf_list))
                    if st_sm.avg_ssm_eachX(curr_x, curr_y, clf_list) > self.threshold:
                        curr_removed_indexList.append(index)

                print('removed list', curr_removed_indexList)
                remove_indexList = remove_indexList + curr_removed_indexList
                # remove noisy data
                X = X.drop(curr_removed_indexList, axis=0)
                Y = Y.drop(curr_removed_indexList, axis=0)
        #print(X)
        #print(Y)
        return X, Y, remove_indexList
