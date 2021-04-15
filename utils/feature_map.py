# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 20:40:17 2021

@author: Stephan
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import utils.mydata
import numpy as np
class feature_map(object):
    """
    Run propagation with one of the image and save the feature map
    """
    def __init__(self, label_tranform, normalize_mode, num_layers, index = 285):
        # get one of the sample from the training set
        batch_image, batch_label, batch_label_ori = utils.mydata.get_one_batch(0, label_tranform, normalize_mode)
        # turn the index into numpy array, this will keep the data size at 1X250X250X3 instead of 250X250X3
        index = np.array([index])
        # sample feature
        self.train_image_sample = batch_image[index,:,:,:]
        # sample labels
        self.train_label_sample = batch_label_ori[index] # [label]
        self.onehot_label_sample = batch_label[index] # [1, 50]
        # number of layers for reference
        self.num_layers = num_layers # number of later
    
    def save_jpg(self, plt, file_path, file_name):
        file_name = file_path + "/" +file_name
        plt.savefig(file_name)
        
    def save_feature_map(self, ConvNet, epoch):
        # organise the save image in folder of epochs
        file_path = "feature_map/" + str(epoch)
        
        # original image
        plt.imshow(self.train_image_sample[0,:,:,0], cmap=mpl.cm.Greys)
        self.save_jpg(plt, file_path, "original.jpg")
        
        # forward propagate to get the feature map of that test image
        ConvNet.Forward_Propagation(self.train_image_sample, self.onehot_label_sample, 'test')
        
        # Feature maps of C1
        print("Feature maps of C1")
        C1map = ConvNet.C1_FP[0]
        fig, axarr = plt.subplots(1,6,figsize=(18,9))        
        for j in range(6):
            axarr[j].axis('off') 
            axarr[j].set_title( 'C1_map'+str(j+1))
            axarr[j].imshow(C1map[:,:,j], cmap=mpl.cm.Greys)
        self.save_jpg(fig, file_path, "C1.jpg")
        
        # Feature maps of S1
        S1map = ConvNet.S1_FP[0]
        print("Feature maps of S1")
        # Feature maps of S1
        fig, axarr = plt.subplots(1,6,figsize=(18,9))
        for j in range(6):
            axarr[j].axis('off') 
            axarr[j].set_title( 'S1_map'+str(j+1))
            axarr[j].imshow(S1map[:,:,j], cmap=mpl.cm.Greys)
        self.save_jpg(fig, file_path, "S1.jpg")
        
        # Feature maps of C2
        C2map = ConvNet.C2_FP[0]
        fig, axarr = plt.subplots(2,8,figsize=(18,5))
        
        for j in range(16):
            x,y = int(j/8), j%8
            axarr[x,y].axis('off') 
            axarr[x,y].set_title( 'C2_map'+str(j+1))
            axarr[x,y].imshow(C2map[:,:,j], cmap=mpl.cm.Greys)
        self.save_jpg(fig, file_path, "C2.jpg")
        
        # Feature maps of S2
        print("Feature maps of S2")
        S2map = ConvNet.S2_FP[0]
        fig, axarr = plt.subplots(2,8,figsize=(18,5))
        
        for j in range(16):
            x,y = int(j/8), j%8
            axarr[x,y].axis('off') 
            axarr[x,y].set_title( 'S2_map'+str(j+1))
            axarr[x,y].imshow(S2map[:,:,j], cmap=mpl.cm.Greys)
        self.save_jpg(fig, file_path, "S2.jpg")
        
        # The 1 more convolutional layer and pooling layer is added in the 9layer version
        if self.num_layers >= 9: 
            # Feature maps of C3
            C3map = ConvNet.C3_FP[0]
            fig, axarr = plt.subplots(3,10,figsize=(18,5))
            
            for j in range(30):
                x,y = int(j/10), j%10
                axarr[x,y].axis('off') 
                axarr[x,y].set_title( 'C3_map'+str(j+1))
                axarr[x,y].imshow(C3map[:,:,j], cmap=mpl.cm.Greys)
            self.save_jpg(fig, file_path, "C3.jpg")
            
            
            # Feature maps of s3
            S3map = ConvNet.S3_FP[0]
            fig, axarr = plt.subplots(3,10,figsize=(18,5))
            
            for j in range(30):
                x,y = int(j/10), j%10
                axarr[x,y].axis('off') 
                axarr[x,y].set_title( 'S3_map'+str(j+1))
                axarr[x,y].imshow(S3map[:,:,j], cmap=mpl.cm.Greys)
            self.save_jpg(fig, file_path, "S3.jpg")