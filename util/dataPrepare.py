# for new_thyroid data
import numpy as np
from sklearn.model_selection import train_test_split
from util.noise_introduction import pairWiseNoise
from pandas import DataFrame
from sklearn import preprocessing

x = np.load("../data/new_thyroid/X.npy")
y = np.load("../data/new_thyroid/y.npy")
#特征归一化
sclaer = preprocessing.MinMaxScaler()
x = sclaer.fit_transform(x)
#特征标准化 (feature standardization)
# x = preprocessing.scale(x)
print(x)
# let the class lable begin with 0
y=y-1
print(y)

min_list = [1,2]
maj_list = [0]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.495)
# print(x_train)
print(y_train)
y_train_df = DataFrame(y_train)
# print(y_train_df)
y_train_noise,mislable_index = pairWiseNoise(y_train_df,maj_list,min_list,0.2)
print(mislable_index)
np.save("../data/new_thyroid/20%noise/x_train.npy",x_train)
np.save("../data/new_thyroid/20%noise/y_train_noise.npy",np.array(y_train_noise))
np.save("../data/new_thyroid/20%noise/x_test.npy",x_test)
np.save("../data/new_thyroid/20%noise/y_test.npy",y_test)
np.save("../data/new_thyroid/20%noise/y_train_mislable_index.npy",mislable_index)
