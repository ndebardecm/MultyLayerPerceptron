#-*-coding: utf-8 -*-

from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import OneHotEncoder
from sklearn import grid_search
import numpy as np
import MLP
reload(MLP)

import numpy as np
import pylab as pl
mnist=fetch_mldata('MNIST original')
Xmnist = mnist.data
Xmnist = Xmnist/255.0 #on met toutes les valeurs entre 0 et 1
Ymnist = mnist.target
encod = OneHotEncoder()
Y = Ymnist.reshape(Ymnist.shape[0], 1)
encod.fit(Y)
Y = encod.transform(Y).toarray()
iperm = np.arange(Xmnist.shape[0])
np.random.shuffle(iperm)
X = Xmnist[iperm]
Y = Y[iperm]
ending_index = 20000
Xtrain = X[0:ending_index]
Ytrain = Y[0:ending_index]
mlp = MLP.MLP([784,150,10], learning_rate=0.35, n_iter=20, auto_update_lr=True)
mlp.fit(Xtrain, Ytrain)
'''
mlp = MLP.MLP()
params = {
          'learning_rate':[0.35, 10]
}
gs = grid_search.GridSearchCV(mlp, params, verbose=1)
#gs = grid_search.GridSearchCV(mlp, params, verbose=1, n_jobs=6)
gs.fit(Xtrain, Ytrain)

print 'grid scores: ', gs.grid_scores_, '\n'
print 'best params: ', gs.best_params_, '\n'


# Sauvegarde du meilleur mlp

from sklearn.externals import joblib

u=gs.best_estimator_.steps[0]
best= u[1]
joblib.dump(best, 'GridSearchBest_MLP.pkl')

'''
print "Score apprentissage de {} % avec un échantillon d'apprentissage comptant {} elements".format(mlp.score(X[0:ending_index],Y[0:ending_index]), ending_index)
print "Score generalisation de {} % ".format(mlp.score(X[ending_index:],Y[ending_index:]))