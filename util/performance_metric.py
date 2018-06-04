from collections import Counter
import math

class Metric(object):
    def gmean(self,y_true,y_predict):
        predcitRightCounter = {}
        for e in set(y_true):
            predcitRightCounter[e] = 0

        lableCounter = {}
        for key, value in Counter(y_true).items():
            lableCounter[key] = value

        for i in range(len(y_true)):
            if y_true[i] == y_predict[i]:
                predcitRightCounter[y_true[i]] = predcitRightCounter[y_true[i]] + 1

        predictAccEachClass = {}
        res = 1
        for key, value in predcitRightCounter.items():
            predictAccEachClass[key] = value / float(lableCounter[key])
            res = res * predictAccEachClass[key]

        return math.sqrt(res)

if __name__ == '__main__':
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 0, 2]
    c = Metric()
    print(c.gmean(y_true,y_pred))


