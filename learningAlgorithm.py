from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import numpy as np

class LearningAlgorithm(object):

    def __init__(self,optimizer,loss,batch_size,epoch,model_mode,inputDim,outputDim=2):
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
        self.epoch = epoch
        self.model = self.getModel(model_mode,inputDim,outputDim)

    def getMLP_ovo(self,inputDim):
        model = Sequential()
        model.add(Dense(32, input_shape=(inputDim,)))  # 输入层
        model.add(Activation('tanh'))  # 激活函数是tanh
        model.add(Dropout(0.5))  # 采用50%的dropout

        model.add(Dense(64))  # 隐藏层节点500个
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))

        model.add(Dense(2))  # one versus one 输出结果是2个类别
        model.add(Activation('softmax'))  # 最后一层用softmax作为激活函数

        return model

    def getMLP_multiclass(self,inputDim,outputDim):
        model = Sequential()
        model.add(Dense(32, input_shape=(inputDim,)))  # 输入层
        model.add(Activation('tanh'))  # 激活函数是tanh
        model.add(Dropout(0.5))  # 采用50%的dropout

        model.add(Dense(64))  # 隐藏层节点500个
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))

        model.add(Dense(outputDim))  # one versus one 输出结果是k个类别
        model.add(Activation('softmax'))  # 最后一层用softmax作为激活函数

        return model
    def getModel(self,model_mode,inputDim,outputDim):
        if model_mode == 'MLP_ovo':
            model = self.getMLP_ovo(inputDim)
            model.compile(optimizer=self.optimizer,loss=self.loss)
            # model.summary()
            return model
        elif model_mode == 'MLP_multiclass':
            model = self.getMLP_multiclass(inputDim,outputDim)
            model.compile(optimizer=self.optimizer,loss=self.loss)
            return model

    def fit(self,X,y,verbose=0):
        # convert integers to dummy variables (one hot encoding)
        y = np_utils.to_categorical(y)
        self.model.fit(X,y,batch_size=self.batch_size,epochs=self.epoch,verbose=verbose)

    def predict(self,X):
        return np.argmax(self.model.predict(X),axis=1)

    def get_weights(self):
        return self.model.get_weights()


