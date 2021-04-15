# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:23:33 2021

@author: Stephan
"""
# Calculation output feautre map size for modified model (9 layers)
# channel
C = 3
# stride S
S = 1
# picture size
W = 250
# filter size F 
F = 3
# zero_pad P (F-1 / 2)
P = 0
# number of filter k
k = 6

#------------------------c1--------------------------
def output_size(W, P, F, S):
    return (W + P * 2  - F)/ S + 1

def parameters(F, C, k): 
    return F**2* C * k + k

def pool_parameters(k): 
    return 2 * k

def size(W, C):
    size = "{} x {} x {}".format(W, W ,C)
    return size

def connections(W, para):
    return W*W*para

C1_size = output_size(W, P, F, S)
para = parameters(F, C, k)
print("C1 shape:", size(C1_size, k))
print("C1 parameters:", para)
print("C1 connection:", connections(C1_size, para))
# max pooling/ average pooling

#---------------------------a1----------------------
# max pooling/ average pooling
# channel
C = k
# stride S
S = 2
# picture size
W = C1_size
# filter size F 
F = 2
# zero_pad P (F-1 / 2)
P = 0
# number of filter k
k = 6

thesize = output_size(W, P, F, S)
para = pool_parameters(k)
print("a1 shape:", size(thesize, C))


# ------------------------C2------------------------
# max pooling/ average pooling
# channel
C = k
# stride S
S = 1
# picture size
W = thesize
# filter size F 
F = 3
# zero_pad P (F-1 / 2)
P = 0
# number of filter k
k = 6

thesize = output_size(W, P, F, S)
para = parameters(F, C, k)
print("C2 shape:", size(thesize, k))
print("C2 parameters:", para)
print("C2 connection:", connections(C1_size, para))

#---------------------------a2----------------------
# max pooling/ average pooling
# channel
C = k
# stride S
S = 2
# picture size
W = thesize
# filter size F 
F = 2
# zero_pad P (F-1 / 2)
P = 0

thesize = output_size(W, P, F, S)
para = pool_parameters(k)
print("a2 shape:", size(thesize, C))

# ------------------------C3------------------------
# max pooling/ average pooling
# channel
C = k
# stride S
S = 2
# picture size
W = thesize
# filter size F 
F = 3
# zero_pad P (F-1 / 2)
P = 0
# number of filter k
k = 16

thesize = output_size(W, P, F, S)
para = parameters(F, C, k)
print("C3 shape:", size(thesize, k))
print("C3 parameters:", para)
print("C3 connection:", connections(C1_size, para))

#---------------------------a3----------------------
# max pooling/ average pooling
# channel
C = k
# stride S
S = 2
# picture size
W = thesize
# filter size F 
F = 2
# zero_pad P (F-1 / 2)
P = 0

thesize = output_size(W, P, F, S)
para = pool_parameters(k)
print("a3 shape:", size(thesize, C))

# ------------------------C4------------------------
# max pooling/ average pooling
# channel
C = k
# stride S
S = 2
# picture size
W = thesize
# filter size F 
F = 3
# zero_pad P (F-1 / 2)
P = 0
# number of filter k
k = 16

thesize = output_size(W, P, F, S)
para = parameters(F, C, k)
print("C4 shape:", size(thesize, k))
print("C4 parameters:", para)
print("C4 connection:", connections(C1_size, para))

#---------------------------a4----------------------
# max pooling/ average pooling
# channel
C = k
# stride S
S = 2
# picture size
W = thesize
# filter size F 
F = 3
# zero_pad P (F-1 / 2)
P = 0

thesize = output_size(W, P, F, S)
para = pool_parameters(k)
print("a4 shape:", size(thesize, C))
'''
# ------------------------C5------------------------
# max pooling/ average pooling
# channel
C = k
# stride S
S = 1
# picture size
W = thesize
# filter size F 
F = 3
# zero_pad P (F-1 / 2)
P = 0
# number of filter k
k = 16

thesize = output_size(W, P, F, S)
para = parameters(F, C, k)
print("C5 shape:", size(thesize, k))
print("C5 parameters:", para)
print("C5 connection:", connections(C1_size, para))

#---------------------------a5----------------------
# max pooling/ average pooling
# channel
C = k
# stride S
S = 2
# picture size
W = thesize
# filter size F 
F = 2
# zero_pad P (F-1 / 2)
P = 0

thesize = output_size(W, P, F, S)
para = pool_parameters(k)
print("a5 shape:", size(thesize, C))
'''