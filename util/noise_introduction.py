# given a pair of classes(y1, y2) and a noise level ρ,
# an instance with label y1 has a probability of ρ to be incorrectly labeled as y2,
# so does an instance with label y2.

import numpy as np
import pandas as pd

# y_true should be the Dataframe
def pairWiseNoise(y_true,maj_list,min_list,p):
    maj_indexList = []
    min_indexList = []

    for maj in maj_list:
        maj_indexList=maj_indexList+(y_true[y_true[0] == maj].index.tolist())
    for min in min_list:
        min_indexList=min_indexList+(y_true[y_true[0] == min].index.tolist())

    maj_random_incorrectly_label_size = int(len(maj_indexList)*p)
    maj_mislable_index = np.random.choice(maj_indexList,maj_random_incorrectly_label_size)

    min_random_incorrectly_label_size = int(len(min_indexList) * p)
    min_mislable_index = np.random.choice(min_indexList, min_random_incorrectly_label_size)


    for maj in maj_mislable_index:
        y_true.loc[[maj]] = np.random.choice(min_list,1,replace=True)

    for min in min_mislable_index:
        y_true.loc[[min]] = np.random.choice(maj_list,1,replace=True)

    mislable_index = np.append(maj_mislable_index,min_mislable_index)

    return y_true,mislable_index

if __name__ == '__main__':
    y = [1,1,1,1,2,2,3]
    y = pd.DataFrame(y)
    maj_list = [1]
    min_list=[2,3]
    p = 0.5
    print(y)
    print(pairWiseNoise(y,maj_list,min_list,p))