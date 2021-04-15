# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 21:16:34 2021

@author: Stephan
"""

import numpy as np
import h5py
import utils.mydata as mydata

# all the data preprocessing function are stored in the file "mydata.py"
fixed_size  = tuple((250,250)) # size for image

# read the reference data filr, column name = ['img_link', 'class']
train_ref, val_ref, test_ref= mydata.readdata("train.txt"), mydata.readdata("val.txt"), mydata.readdata("test.txt") 

train_ref_list = mydata.random_mini_batches_list(train_ref, mini_batch_size = 500, one_batch=False)

# -----------------------For training-------------------#
for i in range(len(train_ref_list)):
    # read the images of entire train data
    image_train, label_train = mydata.import_feature(train_ref_list[i], fixed_size)
    fileName = "data/{}.h5".format(i)
    with h5py.File(fileName, "w") as out:
        out.create_dataset('image_train', data=np.array(image_train))
        out.create_dataset('label_train', data=np.array(label_train))

# -----------------------For validation-------------------#
# read the images of entire train data
image_val, label_val = mydata.import_feature(val_ref, fixed_size)

# -----------------------For testing-------------------#
# read the images of entire train data
image_test, label_test = mydata.import_feature(test_ref, fixed_size)

fileName = 'data.h5'
with h5py.File(fileName, "w") as out:
    # out.create_dataset('image_train', data=np.array(image_train))
    # out.create_dataset('label_train', data=np.array(label_train))
    out.create_dataset('image_val', data=np.array(image_val))
    out.create_dataset('label_val', data=np.array(label_val))
    out.create_dataset('image_test', data=np.array(image_test))
    out.create_dataset('label_test', data=np.array(label_test))


