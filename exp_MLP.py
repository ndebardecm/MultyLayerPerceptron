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
ending_index = 2000
Xtrain = X[0:ending_index]
Ytrain = Y[0:ending_index]
mlp = mlp_instance.MLP([784,100,10], learning_rate=0.35, n_iter=10, auto_update_lr=True)
mlp.fit(Xtrain, Ytrain)
print "Score apprentissage : ",mlp.score(X[ending_index:],Y[ending_index:])