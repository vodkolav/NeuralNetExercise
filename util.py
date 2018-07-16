# -*- coding: utf-8 -*-

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



def softmax(x):
    expx = np.exp(x)
    #np.sum()
    return expx/(expx.sum(axis = 1, keepdims = True))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return x*(x>0)

def tanh(x):
    return np.tanh(x)

def activation(x):
    #return sigmoid(x)
    return relu(x)
    #return tanh(x)
    
def activation_deriv(x):
    #return x*(1-x)  #sigmoid'
    return x>0      #relu'
    #return 1-x**2   #tanh'

#%%
def one_hot(col):
    cat = max( max(col)+1, len(np.unique(col)))
    colhot = np.zeros((col.shape[0], cat))                     
    colhot[np.arange(col.shape[0]),col] = 1
    return np.asarray(colhot)
    #return pd.DataFrame(colhot, dtype = int)




#%%
def clas_rate(Y,P):
    if len(Y.shape) > 1:
        Y = np.argmax(Y, axis = 1)
    if len(P.shape) > 1:
        P = np.argmax(P, axis = 1)
    return np.mean(np.equal(Y,P))
#test it 
#y = [0,1,0,1,1,1,0,1,0,0,1,1,0,1,0,0,1,0,1,0,1,0,1]
#p = [1,0,1,0,1,0,1,0,0,0,1,0,1,1,1,0,1,1,1,0,0,0,0]
if (False):
    y = np.random.randint(0,7, size = 12)
    p = np.random.randint(0,7, size = 12)
    y.shape
    clas_rate(y,p)
    
    
    y = one_hot(y)
    p = one_hot(p)
    y.shape
    clas_rate(y,p)
    
    #%%
    y
    p
    np.equal(y,p)

#%%

def cost(T,Y):
    return np.sum(T * np.log(Y))    


def readData(nr = None):
    return pd.read_csv('fer2013.csv',nrows = nr)


def ShowPic(pix, picnum):
    pic = pix[picnum]
    #pic = np.matrix([int(i) for i in pic.split(' ')])
    pic = pic.reshape((48,48))
    plt.imshow(pic)
    

def ConvertData(pix):
    randrow = pix[np.random.randint(0, len(pix))]
    if (isinstance(randrow, np.ndarray)):
        print('already converted')
        if(randrow.max()<=1):
            print('already normalized')
            return pix, pix
        else:
            return pix/255, pix/255
    converted = np.zeros((len(pix), 2304))
    #converted.shape
    #type(converted)
    for picnum in range((len(pix))): #5): #
        if(picnum %1000 == 0):
            print('Converting image num ', picnum, ' of ', len(pix))
        pic = pix[picnum]
        pic = np.asarray([int(i) for i in pic.split(' ')])
        converted[picnum,:] = pic
    print('finished!')
    return converted/256, converted/256
#pix = fer['pixels']






