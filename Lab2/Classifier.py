#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:20:12 2020

@author: pietro
"""

import numpy as np


#hstack and vstack are helpful to arrange the vectors one on top and one next to one another
# X is the input, Y represents the expected output
X = np.vstack([(np.random.rand(1000,2)*5), (np.random.rand(1000,2)*10)])
Y = np.hstack([([0]*1000), [1]*1000])


# Since we want to use a cross-entropy loss function and a softmax activation function
# a one hot vector is more suited than Y for calculations

Z = np.zeros((2000,2))

# We assign the correct values to Z on the basis of Y

for i in range(2000):
    Z[i,Y[i]] = 1


# Randomly initialize weights
    
W1 = np.random.randn(3,2)
B1 = np.random.randn(3)
W2 = np.random.randn(2,3)
B2 = np.random.randn(2)


def Forward(X, W1, B1, W2, B2):
    #first layer
    
    H = 1/(1+np.exp(-(X.dot(W1.T)+B1)))
    
    #second layer

    A = H.dot(W2.T) + B2
    expA = np.exp(A)
    
    #We use the softmax operator to evaluate our output
    
    Output = expA/expA.sum(axis=1, keepdims = True)
    
    # We return both the final output and the output from the hidden layer
    
    return Output, H

def diff_W2(H, Z, Output): 
    return H.T.dot(Z-Output)

def diff_W1(X,H,Z,Output,W2):
    dZ = (Z-Output).dot(W2)*H*(1-H)
    return X.T.dot(dZ)

def diff_B2(Z,Output):
    return (Z-Output).sum(axis=0)

def diff_B1(Z,Output, W2,H):
    return ((Z-Output).dot(W2)*H*(1-H)).sum(axis=0)

learning_rate = 1e-3

for epoch in range(5000):
    
    Output, H = Forward(X, W1, B1, W2, B2)
    
    W2 += learning_rate * diff_W2(H, Z, Output).T
    B2 += learning_rate*diff_B2(Z, Output)
    W1 += learning_rate * diff_W1(X, H, Z, Output, W2).T
    B1 += learning_rate * diff_B1(Z, Output, W2, H)
    if not epoch % 50:
        
        # Cross entropy function
        
        Error = np.mean(-(Z*np.log(Output)))
        print('Epoch: ', epoch, ' Error: ', Error)
    
    
    
# We use three points to see if the outcome is the expected one: 1 point that very likely belongs to class A, 
# one that surely belongs to class B and one that might belong to both

X_Test1 = 0.01*np.ones([1,2])
X_Test2 = 7.5*np.ones([1,2])
X_Test3 = 4.5*np.ones([1,2])
Y_Test1 , hidden_Tes1= Forward(X_Test1, W1, B1, W2, B2)
Y_Test2 , hidden_Tes2= Forward(X_Test2, W1, B1, W2, B2)
Y_Test3 , hidden_Tes3= Forward(X_Test3, W1, B1, W2, B2)
print('First Test:\n prob of class 0 >>>>>> {} \n prob of class 1 >>>>>> {} \n'.format(Y_Test1[0,0], Y_Test1[0,1]))
print('Second Test:\n prob of class 0 >>>>>> {} \n prob of class 1 >>>>>> {} \n'.format(Y_Test2[0,0], Y_Test2[0,1]))
print('Third Test:\n prob of class 0 >>>>>> {} \n prob of class 1 >>>>>> {} \n'.format(Y_Test3[0,0], Y_Test3[0,1]))