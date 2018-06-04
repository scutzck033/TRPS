import numpy as np

class SSM(object):

    def __init__(self,Q=0.1,beta=50):
        self.Q = Q
        self.beta = beta

    # get the pertubed sample for each training sample
    def getPertubedSample(self,x):
        num_features = len(x)
        pertubedSample = np.array(x, dtype='float')
        for i in range(num_features):
            pertubedSample[i] = float(x[i]) + np.random.uniform(-self.Q, self.Q)
        return pertubedSample

    # get the ssm for each sample through each base classifier
    def ssm_eachX(self,x, y, clf):
        res = 0
        for i in range(self.beta):
            pertubed_sample = np.reshape(self.getPertubedSample(x), (1, -1))
            res += abs(y - clf.predict(pertubed_sample))

        return res / float(self.beta)

    # get the average ssm for each sample yielded by base classifiers
    def avg_ssm_eachX(self,x, y, clf_list):
        res = 0
        for i in range(len(clf_list)):
            res += self.ssm_eachX(x, y, clf_list[i])
        return res / float(len(clf_list))
