# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 22:07:19 2021

@author: Stephan
"""

import numpy as np

def pool_forward(A_prev, hparameters, mode):
    # hparameters_pooling   = {"stride": , "f": }
    # shape of input layer, e.g. batchsize X W X W X channel
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # output height
    n_H = int(1 + (n_H_prev - f) / stride)
    # output weight
    n_W = int(1 + (n_W_prev - f) / stride)
    # output channel
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))      
    for h in range(n_H):                      # loop on the vertical axis of the output volume
        for w in range(n_W):                  # loop on the horizontal axis of the output volume
            # Use the corners to define the current slice on the ith training example of A_prev, channel c
            A_prev_slice = A_prev[:, h*stride:h*stride+f, w*stride:w*stride+f, :]  
            # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. 
            if mode == "max":
                A[:, h, w, :] = np.max(A_prev_slice, axis=(1,2))
            elif mode == "average":
                A[:, h, w, :] = np.average(A_prev_slice, axis=(1,2))

    cache = (A_prev, hparameters)
    assert(A.shape == (m, n_H, n_W, n_C))
    # A = output layer of the pooling, cache = input layer for backward propagation
    return A, cache

def pool_backward(dA, cache, mode):
    """
    Implements the backward pass of the pooling layer
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    (A_prev, hparameters) = cache
    
    stride = hparameters["stride"]
    f = hparameters["f"]

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape #256,28,28,6
    m, n_H, n_W, n_C = dA.shape                    #256,14,14,6
    
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev)) #256,28,28,6
        
    for h in range(n_H):                    # loop on the vertical axis
        for w in range(n_W):                # loop on the horizontal axis
            # Find the corners of the current "slice"
            vert_start, horiz_start  = h*stride, w*stride
            vert_end,   horiz_end    = vert_start+f, horiz_start+f
            
            # Compute the backward propagation in both modes.
            if mode == "max":
                A_prev_slice = A_prev[:, vert_start: vert_end, horiz_start: horiz_end, :] 
                A_prev_slice = np.transpose(A_prev_slice, (1,2,3,0))
                mask = A_prev_slice==A_prev_slice.max((0,1))           
                mask = np.transpose(mask, (3,2,0,1))                   
                dA_prev[:, vert_start: vert_end, horiz_start: horiz_end, :] \
                      += np.transpose(np.multiply(dA[:, h, w, :][:,:,np.newaxis,np.newaxis],mask), (0,2,3,1))

            elif mode == "average":
                da = dA[:, h, w, :][:,np.newaxis,np.newaxis,:]  
                dA_prev[:, vert_start: vert_end, horiz_start: horiz_end, :] += np.repeat(np.repeat(da, f, axis=1), f, axis=2)/f/f
    
    assert(dA_prev.shape == A_prev.shape)
    return dA_prev
################################### functions below are NOT USED in training ################################################
def subsampling_forward(A_prev, weight, b, hparameters):
    A_, cache = pool_forward(A_prev, hparameters, 'average') 
    A = A_ * weight + b
    cache_A = (cache, A_)
    return A, cache_A

def subsampling_backward(dA, weight, b, cache_A_):
    (cache, A_) = cache_A_
    db = dA
    dW = np.sum(np.multiply(dA, A_))
    dA_ = dA * weight
    dA_prev = pool_backward(dA_, cache, 'average') 
    return dA_prev, dW, db