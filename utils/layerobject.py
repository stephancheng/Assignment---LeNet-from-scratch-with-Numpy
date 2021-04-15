# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 21:18:06 2021

@author: Stephan
"""

import numpy as np
from utils.conv_operations import conv_forward, conv_backward, conv_SDLM
from utils.Pooling_util import pool_forward, pool_backward, subsampling_forward, subsampling_backward
import utils.activations as acts
from utils.util_fun import *

class ConvLayer(object):
    def __init__(self, kernel_shape, hparameters, init_mode='Gaussian_dist'):
        """
        kernel_shape: (n_f, n_f, n_C_prev, n_C)
        hparameters = {"stride": s, "pad": p}
        """
        self.hparameters = hparameters
        # inititalize weight with gaussain distribution
        self.weight, self.bias = initialize(kernel_shape, init_mode)
        self.v_w, self.v_b = np.zeros(kernel_shape), np.zeros((1,1,1,kernel_shape[-1]))
        
    def foward_prop(self, input_map):
        # output_map = output layer of the pooling, cache = input layer for backward propagation
        output_map, self.cache = conv_forward(input_map, self.weight, self.bias, self.hparameters)
        return output_map
    
    def back_prop(self, dZ, momentum, weight_decay):
        dA_prev, dW, db = conv_backward(dZ, self.cache)
        self.weight, self.bias, self.v_w, self.v_b = \
            update(self.weight, self.bias, dW, db, self.v_w, self.v_b, self.lr, momentum, weight_decay)
        return dA_prev  
  
    def learning(self, lr_global):
        self.lr = lr_global       
        
class PoolingLayer(object):
    def __init__(self, hparameters, mode):
        # hparameters_pooling   = {"stride": , "f": } 
        self.hparameters = hparameters
        # mode = "average" or "max"
        self.mode = mode
        
    def foward_prop(self, input_map):   # n,84,84,6 / n,10,10,16
        # A = out
        A, self.cache = pool_forward(input_map, self.hparameters, self.mode)
        return A
    
    def back_prop(self, dA):
        dA_prev = pool_backward(dA, self.cache, self.mode)
        return dA_prev
    
    def learning(self, lr_global):
        self.lr = lr_global       

class Activation(object):
    def __init__(self, mode): 
        # The activation function are stored in activation.py, import as "acts"
        # forward propagate object, backward propagate object, list of activation (In the same order)
        (act, d_act), actfName = acts.activation_func()
        # Look for the index of the specific activation function in the list
        act_index  = actfName.index(mode)
        # forward propagate function
        self.act   = act[act_index]
        # backward propagate function
        self.d_act = d_act[act_index]
        
    def foward_prop(self, input_image): 
        self.input_image = input_image
        return self.act(input_image)
    
    def back_prop(self, dZ):
        dA = np.multiply(dZ, self.d_act(self.input_image)) 
        return dA
    
class FC(object):
    """
    Fully connected layer
    """
    def __init__(self, weight_shape, init_mode='Gaussian_dist'):
        
        self.cache = None
        # initialise weights and historical weights
        self.v_w, self.v_b = np.zeros(weight_shape), np.zeros((weight_shape[-1],))
        self.weight, self.bias = initialize(weight_shape, init_mode)
        
    def foward_prop(self, X): # x = minibatch size X 120
        # out = WX+b
        out = np.dot(X, self.weight) + self.bias
        # save input for backpropagation
        self.cache = X
        return out

    def back_prop(self, dZ, momentum, weight_decay):
        X = self.cache
        dX = np.dot(dZ, self.weight.T).reshape(X.shape)
        dW = np.dot(X.reshape(X.shape[0], np.prod(X.shape[1:])).T, dZ)
        db = np.sum(dZ, axis=0)
        self.weight, self.bias, self.v_w, self.v_b = \
            update(self.weight, self.bias, dW, db, self.v_w, self.v_b, self.lr, momentum, weight_decay)
        return dX
    
    def learning(self, lr_global):
        self.lr = lr_global      
        
class SMLayer(object):       
    def foward_prop(self, input_array, label, mode):
        self.softM = acts.softmax(input_array) 
        loss = self.softM.softmax_loss(label)
        if mode == 'train':    
            return loss
        elif mode == 'test':
            # output the probability of class of each sample
            softmax_score = self.softM.forward() 
            # list of top predicted class of each sample
            result_top1 = np.argmax(softmax_score, axis=1)
            return result_top1
        
    def back_prop(self):
        dA = self.softM.backward() # [n, class]
        return dA

    def learning(self, lr_global):
        self.lr = lr_global    