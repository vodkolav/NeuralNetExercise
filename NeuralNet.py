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
import glob
#%matplotlib inline
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"

from util import *


#%% Layer class
class nnet_layer:
    #W
    #b
    #Z
    def __init__(self,incoming,outcoming):
        self.W = np.random.randn(incoming,outcoming)/np.sqrt(incoming + outcoming)
        self.b = np.zeros((1,outcoming))  #or random.rand  
    @property
    def W(self):
        return self.__W
    
    @W.setter
    def W(self, W):
        self.__W = W    
        
    @property
    def b(self):
        return self.__b    
    
    @b.setter
    def b(self, b):        
        self.__b = b
        
    @property
    def Z(self):
        return self.__Z   
    
    @Z.setter
    def Z(self, Z):        
        self.__Z = Z
        
    @property
    def indx(self):
        return self.__indx   
    
    @indx.setter
    def indx(self, indx):        
        self.__indx = indx
        
        
#%% NeuralNet class        
class NeuralNet():
    
#%% Properties
    
    @property
    def L(self):
        return self.__layers
    @L.setter
    def L(self, incoming):        
        self.__layers = incoming
        
        
    @property
    def Epochs(self):
        return self.__Epochs
    @Epochs.setter
    def Epochs(self, incoming):        
        self.__Epochs = incoming    
        
        
    @property
    def learning_rate(self):
        return self.__learning_rate
    @learning_rate.setter
    def learning_rate(self, incoming):        
        self.__learning_rate = incoming    
        
        
    @property
    def verbose(self):
        return self.__verbose    
    @verbose.setter
    def verbose(self, incoming):        
        self.__verbose = incoming
    
    
    @property
    def batch_size(self):
        return self.__batch_size
    
    @batch_size.setter
    def batch_size(self, incoming):        
        self.__batch_size = incoming
               
        
#%% Functions        
    def __init__(self, Epochs = 1000, learning_rate = 10e-5, verbose = False):
        self.__layers = []
        self.costs = []
        self.__Epochs = Epochs
        self.__learning_rate = learning_rate
        self.__verbose = False
        
#    def __init__(self, *args, **kwargs):
#        if len(kwargs)==1:
#            self.load(kwargs.get()
        
        
    def save(self, path = "" ):
        ###Save the nnet to disk
        # nnet = {'X': X,'W1': W1,'b1': b1, 'W2': W2, 'b2': b2}   
        
        if (path ==""):
            path ='nnet' + time.strftime("%Y%m%d") + '.pkl'
        print('saving to ',path)
        with open(path, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(self.L, f)
            
    def load(self, path = ""):
        #Load the nnet from disk
        if (path == ""):
            files = glob.glob('*.pkl')
            filename  = files[-1]
        else:
            filename = path
        #filename ='nnet' + time.strftime("%Y%m%d") + '.pkl'
        print(filename)
        with open(filename, 'rb') as f:  # Python 3: open(..., 'rb')
            self.L = pickle.load(f)
#            tmp_dict= pickle.load(f)
#            self.__dict__.update(tmp_dict) 
        
    def add_layer(self,layer):
        layer.indx = len(self.L)+1
        self.L.append(layer)        
    
        
    def nnet_predict(X,W1,b1,W2,b2):     
        Z = activation(X.dot(W1) + b1 )     
        Yhat = softmax(Z.dot(W2) + b2)   
        return Yhat, Z

    def predict(self, X):     
        x = X.copy()
        for l in self.L:

            Z = X.dot(l.W) + l.b            
            
               
            if l.indx< len(self.L):
                X = activation(Z)           
                l.Z = Z
            else:        
                Yhat = softmax(Z)   
                l.Z = x
                #print("Yhat: ", Yhat.shape)    
                  
            if self.verbose:
                print("layer ", l.indx, " ==========")
                print("X: ", X.shape)                           
                print("W: ", l.W.shape)
                print("b: ", l.b.shape)
                print("Z: ", l.Z.shape)      
            
        return Yhat
    
    
    
    def train(self,X,T):
        
        t = time.time()
        for epoch in range(self.Epochs):
    #         print('X:',X.shape)
    #         print('Y:',Y.shape)
    #         print('T:',T.shape)
    #         print('W1:',W1.shape)
    #         print('b1:',b1.shape)
    #         print('W2:',W2.shape)
    #         print('b2:',b2.shape)
            out = self.predict(X)
    #         print('out: ', out.shape)
    #         print('hid: ', hid.shape)
    #        print(out)
            if epoch %1000 ==0:
                c = cost(T,out)
                P = np.argmax(out, axis = 1)                
                r = clas_rate(T,P)
                print("1000xEpochs:", "{:5.0f}".format(epoch/1000) , "|cost: ", "{:5.3f}".format(c), '|Classification rate: ', "{:5.3f}".format(r))
                self.costs.append(c)

            for l in range(len(self.L)-1, -1,-1) : # range(len(self.layers)):  
                   
                
                W,b = self.deriv_W_b(X, T, out, l)
                
                
                #print("W: ", W.shape, "b: ", b.shape)
                
                #l.W,l.b = np.add([l.W,l.b],np.multiply(self.learning_rate, (W,b)))  
                
                #print("self.L[l].W",self.L[l].W.shape,"W",W.shape)
                
                self.L[l].W = np.add(self.L[l].W,np.multiply(self.learning_rate, W))
                
                self.L[l].b = np.add(self.L[l].b,np.multiply(self.learning_rate, b))
                
              
                #print("out: ", out.shape, "l: ", l.indx)
                #self.layers[l.indx-1] = l
                
#                 W2,b2 = np.add([W2,b2],np.multiply(learning_rate, derivative_w2_b2(hid, T, out)))
#                 W1,b1 = np.add([W1,b1],np.multiply(learning_rate, derivative_w1_b1(X, hid, T, out, W2))) 
                

        plt.plot(self.costs)
        plt.show() 
        print('elapsed: %.2f sec' %  (time.time() - t))
        return 

    def  deriv_W_b(self, X, T, Y, l):
        TYWdZ = T - Y
    
        #print("X: ", X.shape,  "TYWdZ", TYWdZ.shape, "W: ", self.L[l].W.shape, "Z: ", self.L[l].Z.shape)# , "dZ: ", activation_deriv(self.L[ll].Z).shape )
        last = len(self.L)-1
        for ll in range(last,l,-1):
            #print("from, to, current: " ,last, l,ll)
            #print("TYWdZ: ", TYWdZ.shape,"self.L[ll].W" , self.L[ll].W.shape , "self.L[ll-1].Z", self.L[ll-1].Z.shape)
            
            
            TYWdZ = TYWdZ.dot(self.L[ll].W.T)*activation_deriv(self.L[ll-1].Z)
            
        #TmY = TmY.dot(WxdZ)               
        #print("self.L[l.indx-2].Z" , self.L[l.indx-2].Z.shape ,"TYWdZ: ", TYWdZ.shape)
        dW = self.L[l-1].Z.T.dot(TYWdZ)
        db = TYWdZ.sum(axis = 0)
        #print("dW: ", dW.shape,  "TYWdZ", TYWdZ.shape, "W: ", self.L[l].W.shape, "Z: ", self.L[l].Z.shape)
        #self.L[l].W.shape = dW.shape
        return dW, db 

    def  derivative_w1_b1(X, Z, T, Y, W2):
        dZ = (T - Y).dot(W2.T)*activation_deriv(Z)
        return X.T.dot(dZ), dZ.sum(axis = 0)
    
    def  derivative_w2_b2(Z, T, Y):
        TmY = T - Y
        return (Z.T).dot(TmY), TmY.sum(axis = 0)
    
    
    
    
    