#%%
import os 
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from util import *
from NeuralNet import *

#%%


if  0:  # Recreate train, test sets
    
    if not 'fer' in locals():
        fer = readData()#  nr = 8000)#  
        pix = fer['pixels']    
        X, pix = ConvertData(pix) #normalizing
    T_cold = np.asarray([int(i) for i in fer['emotion']])
    
    Xdisg = X[T_cold==1].repeat(9, axis = 0)
    Tdisg =  T_cold[T_cold == 1].repeat(9, axis = 0)
    
    X = np.vstack((X,Xdisg))
    T_cold = np.hstack((T_cold,Tdisg))
    
    T = one_hot(T_cold)        
    X.shape, X.mean(), X.max(), X.std()
    T.shape
    X,T = shuffle(X,T)
    #first, last = (1000, 4000)
    first, last = (0, len(X))
    whichsamples = range(first, last)    
    print('using samples: ', whichsamples)
    X_train, X_test, T_train, T_test = train_test_split(X[whichsamples,:], T[whichsamples,:], test_size=0.3)
    restart = False
     
     
#%%
if 0:
  emolabel = np.array(['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])
  ylab = emolabel[T_cold]
  ylab.shape
  ylab      
  sns.countplot(ylab)      
  from collections import Counter
  Counter(ylab)  
  
  Tdisg.shape
    #%%
#if 'restart' in locals() and restart:
#    model = NeuralNet(Epochs= 10000, learning_rate= 1e-5)
#    model.add_layer(nnet_layer(2304, 20))
#    model.add_layer(nnet_layer(20, 30))
#    model.add_layer(nnet_layer(30, 15))
#    model.add_layer(nnet_layer(15, 7))
#    model.layers[0].indx
#  
    
if 0:
  model.save("neuralNetModel.pkl")    
    
#%% Model creation
    
if 1:
  model = NeuralNet(Epochs= 1e4, learning_rate= 1e-6, batch_size = 10000,  verbose = False, regularisation = 1e-9)
  model.add_layer(nnet_layer(2304, 200))
  model.add_layer(nnet_layer(200, 7))
  refit = False
else:
  model.Epochs = 1e4
  model.learning_rate = 1e-6
  model.batch_size = 10000
  model.verbose = False
  model.regularisation = 1e-9
  
#    model.Epochs = 100000
#    model.learning_rate = 1e-6
#   
#%%  Training
if 1:
  
  model.train(X_train, T_train)



#%% Predicting and testing
if 1:
    model.verbose = True
    Y_pred = model.predict(X_test)
    print( 'classification rate: ', clas_rate(Y_pred, T_test))
    cm = confusion_matrix(np.argmax(Y_pred, axis = 1),np.argmax(T_test, axis = 1))
    plt.figure()
    plt.imshow(cm)
#sns.countplot(Y_dev)
# plt.figure()
# sns.countplot(T_dev.argmax(axis = 1))    

