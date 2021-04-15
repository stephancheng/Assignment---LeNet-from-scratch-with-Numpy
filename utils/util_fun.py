# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 21:20:57 2021

@author: Stephan
"""

import numpy as np

# padding for the matrix of images
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, ), (pad, ), (pad, ), (0, )), 'constant', constant_values=(0, 0))    
    return X_pad

# initialization of the weights & bias
def initialize(kernel_shape, mode='Fan-in'):
    # kernel_share = [n_f, n_f, n_C_prev, n_C]
    b_shape = (1,1,1,kernel_shape[-1]) if len(kernel_shape)==4 else (kernel_shape[-1],)
    
    if mode == 'Xavier':
        if len(kernel_shape)==4:
            # Din is kernel_size2 * input_channels
            Din = kernel_shape[0] * kernel_shape[1] * kernel_shape[2]
            weight = np.random.randn(kernel_shape[0], kernel_shape[1], kernel_shape[2],kernel_shape[3])/ np.sqrt(Din)
        else: 
            Din = kernel_shape[0]   
            weight = np.random.randn(kernel_shape[0], kernel_shape[1])/ np.sqrt(Din)
        bias   = np.ones(b_shape)*0.01
    
    elif mode == 'Gaussian_dist':
        mu, sigma = 0, 0.1
        weight = np.random.normal(mu, sigma,  kernel_shape) 
        bias   = np.ones(b_shape)*0.01
        
    elif mode == 'Fan-in': #original init. in the paper
        Fi = np.prod(kernel_shape)/kernel_shape[-1]
        weight = np.random.uniform(-2.4/Fi, 2.4/Fi, kernel_shape)    
        bias   = np.ones(b_shape)*0.01
    return weight, bias

# update for the weights
def update(weight, bias, dW, db, vw, vb, lr, momentum=0, weight_decay=0):    
    vw_u = momentum*vw - weight_decay*lr*weight - lr*dW
    vb_u = momentum*vb - weight_decay*lr*bias   - lr*db
    weight_u = weight + vw_u
    bias_u   = bias   + vb_u
    return weight_u, bias_u, vw_u, vb_u 

#---------------------------------------------------------------------#
'''
given the model prediction
compare with test label
'''
def evaluate(predictions, test_labels, mode = 'mean'):
    '''   
    Parameters
    ----------
    predictions : TYPE np array
        List of top one prediction by the model.
    test_labels : TYPE np array
        List of true class label of the test data.
    mode : TYPE str
        default = 'mean'    
        mean(average accuracy) or total(total number of correct case).
    Returns
    -------
    accuracy_top1 : TYPE float in mean, int in total
        percentage of top 1 accuracy in mean and total number of matched prediction in total.
    '''
    accuracy_top1 = np.count_nonzero(predictions == test_labels)
    
    if mode == 'total':
        return accuracy_top1
    elif mode == 'mean':
        return accuracy_top1/test_labels.shape[0]*100
    else: 
        print("warning: the mode is wrong, returning mean accuracy")
        return accuracy_top1/test_labels.shape[0]*100