# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 21:40:04 2021

@author: Stephan
"""

# activations
import numpy as np


#Squashing function used in LeNet-5
def LeNet5_squash(x):
    return 1.7159*np.tanh(2*x/3)
def d_LeNet5_squash(x):
    return 1.14393*(1-np.power(tanh(2*x/3),2))
def d2_LeNet5_squash(x):
    return -1.52524*((tanh(2/3*x)))*(1-np.power(tanh(2/3*x),2))

#sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def d_sigmoid(x):
    return np.exp(-x) / np.power((1+np.exp(-x)),2)
def d2_sigmoid(x):
    return 2*np.exp(-2*x)/np.power(np.exp(-x)+1,3)  - np.exp(-x) / np.power((1+np.exp(-x)),2)

#tanh
def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1/np.power(np.cosh(x),2)
def d2_tanh(x):
    return -2*(tanh(x))/np.power(np.cosh(x),2)

#ReLU
def ReLU(x):
    return np.where(x>0, x, 0)
def d_ReLU(x):
    return np.where(x>0, 1, 0)
def d2_ReLU(x):
    return np.zeros(d_ReLU(x).shape)

alpha = {"prelu":0.1, "elu":0.5}
#Parametric ReLU (alpha=0.01 equals to Leaky ReLU)
def PReLU(x, a=alpha["prelu"]):
    return np.where(x>0, x, a*x)
def d_PReLU(x, a=alpha["prelu"]):
    return np.where(x>0, 1, a)
def d2_PReLU(x, a=alpha["prelu"]):
    return np.zeros(d_PReLU(x, a).shape)

#ELU
def ELU(x, a=alpha["elu"]):
    return np.where(x > 0, x, a*(np.exp(x) - 1))
def d_ELU(x, a=alpha["elu"]):
    return np.where(x > 0, 1, ELU(x, a)+a)
def d2_ELU(x, a=alpha["elu"]):
    return np.where(x > 0, 0, ELU(x, a)+a)

def activation_func():
    # list of activation function 
    actf = [LeNet5_squash, sigmoid, tanh, ReLU, PReLU, ELU]
    # forward propagation for each activation function
    actfName = [act.__name__ for act in actf]    
    d_actf = [d_LeNet5_squash, d_sigmoid, d_tanh, d_ReLU, d_PReLU, d_ELU]
    # backward propagation for each activation function
    d_actfName = [d_act.__name__ for d_act in d_actf]
    return (actf, d_actf), actfName

class softmax():
    def __init__(self,A):
        self.A = A
        
    def forward(self):
        try: expA = np.exp(self.A)
        except TypeError:
            expA =  np.exp(self.A.astype(np.float64))
        try: 
            expA = expA.to_numpy()
        except: AttributeError
        base = expA.sum(axis=1, keepdims = True)
        # handle devide by 0 error
        self.softmax_prob = np.divide(expA, base , out=np.zeros(expA.shape, dtype=float), where=base!=0)
        return self.softmax_prob
    
    def softmax_loss(self, y_true):
        self.y_true = y_true
        loss = - self.y_true * np.log(self.forward())
        try: 
            total_loss = loss.sum().sum() # for pandas
        except AttributeError:
            total_loss = loss.sum() # for np array
        return total_loss
    
    def backward(self):
        return self.softmax_prob - self.y_true
    
    

