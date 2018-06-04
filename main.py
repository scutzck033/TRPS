# hyper-parameter:
# beta: number of perturbed samples around each training sample
# Q: the maximum magnitude of perturbation to each feature
# threshold: the value which determines the sample as noise or not
# num_of_baseClf: number of base classifiers
# learningAlgorithmRebalance: which kind of classifier you use to filtrate noisy data
# learningAlgorithmFinal: which kind of classifier you use to be the final classifier
                                           #trained on noise-filtered rebalanced data

import numpy as np
import pandas as pd
from sklearn import svm
from imblearn.over_sampling import SMOTE
from collections import Counter
import ssm_based_noise_filter
from keras.optimizers import SGD,nadam
from collections import Counter
import learningAlgorithm
from util.performance_metric import Metric
from util.visualization import Visualization

if __name__ == '__main__':
# data preparation
    data_name = "new_thyroid_20%noise"

    x_train = np.load("./data/new_thyroid/20%noise/x_train.npy")
    y_train_noise = np.load("./data/new_thyroid/20%noise/y_train_noise.npy")
    noise_index = np.load("./data/new_thyroid/20%noise/y_train_mislable_index.npy")

    x_train = pd.DataFrame(x_train)
    y_train_noise = pd.DataFrame(y_train_noise)

    print(np.shape(x_train))
    print(np.shape(y_train_noise))

# hyperparemeter setting
    inputDim = x_train.shape[1]
    num_class = len(set(y_train_noise[0]))

    num_of_baseClf = 5
    threshold = 0.5
    Q=0.1
    beta=50
    L_rebalance_attributes = {
        'optimizer': nadam(),
        'loss': 'categorical_crossentropy',
        'batch_size': 2,
        'epoch': 10,
        'model_mode': 'MLP_ovo',
        'inputDim': inputDim
    }

    learningAlgorithmfFinal = learningAlgorithm.LearningAlgorithm(optimizer=nadam(),
                            loss='categorical_crossentropy',batch_size=16,
                                            epoch=50,model_mode='MLP_multiclass',
                                                inputDim=inputDim,outputDim=num_class)


    learningAlgorithmfFinalRebalancedRaw = learningAlgorithm.LearningAlgorithm(optimizer=nadam(),
                            loss='categorical_crossentropy',batch_size=16,
                                            epoch=50,model_mode='MLP_multiclass',
                                                inputDim=inputDim,outputDim=num_class)

    learningAlgorithmfFinalRaw = learningAlgorithm.LearningAlgorithm(optimizer=nadam(),
                                                                   loss='categorical_crossentropy', batch_size=16,
                                                                   epoch=50, model_mode='MLP_multiclass',
                                                                   inputDim=inputDim, outputDim=num_class)


#filtrate noisy data
    ssmBasedNoiseFilter = ssm_based_noise_filter.NoiseFilter(num_of_baseClf,threshold,Q,beta)
    x_filtered,y_filtered,filtered_index =ssmBasedNoiseFilter.noiseFilter((x_train,y_train_noise),L_rebalance_attributes)
    np.save('./noise-filtered_data/'+data_name+'x_filtered.npy',x_filtered)
    np.save('./noise-filtered_data/'+data_name+'y_filtered.npy',y_filtered)
    print('filter done!\n')



    x_filtered=np.load('./noise-filtered_data/'+data_name+'x_filtered.npy')
    y_filtered=np.load('./noise-filtered_data/'+data_name+'y_filtered.npy')
    print(x_filtered.shape)
    print(y_filtered.shape)

# rebalance noise-filtered data using smote
    print('rebalance noise-filtered data using smote')
    sm = SMOTE(k_neighbors=3)
    x_sm,y_sm = sm.fit_sample(x_filtered,y_filtered)

    x_sm_compare,y_sm_compare = sm.fit_sample(x_train,y_train_noise)

    print('raw dataset shape {}'.format(Counter(y_train_noise[0])))
    print('Resampled raw dataset shape {}'.format(Counter(y_sm_compare)))
    print('Resampled filtered dataset shape {}'.format(Counter(y_sm)))

# data visualization
    # visualize raw data
    visual_raw = Visualization(x_train, y_train_noise[0],title="raw_data")
    visual_raw.manifold_visualization(path="./visualization/" + data_name[0:-9] + "/",
                                               fileName="raw_data",
                                               save=False)
    # visualize raw rebalanced data
    visual_rebalance = Visualization(x_sm_compare, y_sm_compare,title="raw_rebalanced_data")
    visual_rebalance.manifold_visualization(path="./visualization/" + data_name[0:-9] + "/",
                                            fileName="raw_rebalanced_data",
                                            save=False)
    # visualize filered data
    print(type(y_filtered))
    print(y_filtered.shape)
    visual_filered = Visualization(x_filtered,pd.DataFrame(y_filtered)[0],title="filtered_data")
    visual_filered.manifold_visualization(path="./visualization/" + data_name[0:-9] + "/",
                                          fileName="filtered_data",
                                          save=False)
    # visualize filered rebalanced data
    visual_filered_rebalance = Visualization(x_sm, y_sm,title="filtered_rebalanced_data")
    visual_filered_rebalance.manifold_visualization(path="./visualization/" + data_name[0:-9] + "/",
                                                    fileName="filtered_rebalanced_data",
                                                    save=False)


# train a final classifier using learningAlgorithmFinal
# on the noise-filtered rebalanced dataset
    learningAlgorithmfFinal.fit(x_sm,y_sm,verbose=2)
    learningAlgorithmfFinalRebalancedRaw.fit(x_sm_compare, y_sm_compare)
    learningAlgorithmfFinalRaw.fit(x_train, y_train_noise[0])

# comparative experiment
    x_test = np.load("./data/new_thyroid/20%noise/x_test.npy")
    y_test = np.load("./data/new_thyroid/20%noise/y_test.npy")

    y_pred=learningAlgorithmfFinal.predict(x_test)
    y_pred1 = learningAlgorithmfFinalRebalancedRaw.predict(x_test)
    y_pred2 = learningAlgorithmfFinalRaw.predict(x_test)

    metric = Metric()
    print("raw gmean is: " + str(metric.gmean(y_test, y_pred2)))
    print("smote gmean is: " + str(metric.gmean(y_test, y_pred1)))
    print("ours gmean is: "+str(metric.gmean(y_test,y_pred)))

    print("percentage of correctly founded noise:"+
          str(float(len(set(noise_index)&set(filtered_index)))/len(noise_index)))

#心得体会：
# 1.采用有放回抽样，
# 如果噪声样本的比例比较大时，
# 则可能使得学习的分类器错把正常样本当成噪声样本
# （事实上分类器的训练样本便是以噪声样本为主）
#
# PS：
# 有放回抽样通常会造成过拟合问题
# 这样的话其实也会存在这么一种可能：
# 即使分类器是在正常样本上训练而来，
# 但是其仍然可能将另一部分没见过的正常样本错分
#
# 解决办法：
# 增加基分类器的数目，即采用有放回抽样多次
#
# 2.用分类器判断异常点的话，可能存在这么一种情况：
# 分类器可以正确地分对样本标签，
# 但是事实上某些样本是离群点，
# 那么一旦采用上采样的方法后，
# 则有引入额外噪声的可能。

#解决办法：
# ovo中，不采用有放回采样，
# 而是通过对每个类进行聚类的办法，挑选前k个具有代表性的样本，对分类器进行学习???
# ---不行：聚类方法本身对噪声很敏感