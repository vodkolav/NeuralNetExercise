#%%
import pandas as pd
import numpy as np
import seaborn as sns
import time
import pickle
import timeit
import tensorflow as tf
import keras as ks
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as a3d
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
#%matplotlib inline
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from util import *
from NeuralNet import *

#%%


if  restart:
    
    if not 'fer' in locals():
        fer = readData(nr = 8000)#  )#  
        pix = fer['pixels']    
        X, pix = ConvertData(pix) #normalizing
        T_cold = np.asarray([int(i) for i in fer['emotion']])
        T = one_hot(T_cold)
        X.shape, X.mean(), X.max(), X.std()
        T.shape
    first, last = (1000, 4000)
    whichsamples = range(first, last)    
    print('using samples: ', whichsamples)
    X_train, X_test, T_train, T_test = train_test_split(X[whichsamples,:], T[whichsamples,:], test_size=0.3)
    restart = False
     
     
    #%%
#if 'restart' in locals() and restart:
#    model = NeuralNet(Epochs= 10000, learning_rate= 1e-5)
#    model.add_layer(nnet_layer(2304, 20))
#    model.add_layer(nnet_layer(20, 30))
#    model.add_layer(nnet_layer(30, 15))
#    model.add_layer(nnet_layer(15, 7))
#    model.layers[0].indx
#  
    
#%%
    

model = NeuralNet(Epochs= 100000, learning_rate= 1e-6)
model.add_layer(nnet_layer(2304, 20))
model.add_layer(nnet_layer(20, 7))
refit = False
#else:
#   
#    model.Epochs = 100000
#    model.learning_rate = 1e-6
#   
#%%
model.verbose = False

#model.train(X_train, T_train)



#%%
if False:
    model.verbose = True
    Y_pred = model.predict(X_test)
    print( 'classification rate: ', clas_rate(Y_pred, T_test))
#sns.countplot(Y_dev)
# plt.figure()
# sns.countplot(T_dev.argmax(axis = 1))    

model.save("neuralNetModel.pkl")
