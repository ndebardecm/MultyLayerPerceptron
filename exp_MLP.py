#-*-coding: utf-8 -*-

import sklearn as sk
from sklearn.datasets import fetch_mldata
import MLP as mlp_instance
reload(mlp_instance)

import numpy as np
import pylab as pl
mnist=fetch_mldata('MNIST original')
Xmnist = mnist.data
Xmnist = Xmnist/255.0 #on met toutes les valeurs entre 0 et 1
Ymnist = mnist.target
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
encod = OneHotEncoder()
Y = Ymnist.reshape(Ymnist.shape[0], 1)
encod.fit(Y)
Y = encod.transform(Y).toarray()
iperm = np.arange(Xmnist.shape[0])
np.random.shuffle(iperm)
X = Xmnist[iperm]
Y = Y[iperm]
Xtrain = X[0:100]
Ytrain = Y[0:100]
mlp = mlp_instance.MLP([784,500,10], learning_rate=0.1, n_iter=100)
mlp.fit(Xtrain, Ytrain)
