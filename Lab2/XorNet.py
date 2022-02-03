#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:20:12 2020

@author: pietro
"""

import numpy as np



X = np.random.randint(2, size=[50,2])
Z = np.array([X[:,0] ^ X[:, 1]]).T

    
W1 = np.random.randn(3,2)
B1 = np.random.randn(3)
W2 = np.random.randn(1,3)
B2 = np.random.randn(1)

    
def sigm(X, W, B):
    
    M = 1/(1+np.exp(-(X.dot(W.T)+B)))

    return M

def Forward(X, W1, B1, W2, B2):
    #first layer
    
    H = sigm(X,W1,B1)
    
    #second layer
    
    Y = sigm(H,W2,B2)
    
    # We return both the final output and the output from the hidden layer

    return Y, H




def diff_B2(Z,Y):
    dB = (Z-Y)*Y*(1-Y)
    return dB.sum(axis=0)

def diff_W2(H, Z, Y): 
    dW = (Z-Y)*Y*(1-Y)
    return H.T.dot(dW)

def diff_W1(X,H,Z,Y,W2):
    dZ = (Z-Y).dot(W2)*Y*(1-Y)*H*(1-H)
    return X.T.dot(dZ)

def diff_B1(Z,Y, W2,H):
    return ((Z-Y).dot(W2)*Y*(1-Y)*H*(1-H)).sum(axis=0)

learning_rate = 1e-2

for epoch in range(10000):
    
    Y, H = Forward(X, W1, B1, W2, B2)
    
    W2 += learning_rate * diff_W2(H, Z, Y).T
    B2 += learning_rate*diff_B2(Z, Y)
    W1 += learning_rate * diff_W1(X, H, Z, Y, W2).T
    B1 += learning_rate * diff_B1(Z, Y, W2, H)
    if not epoch % 50:
        Accuracy = 1 - np.mean((Z - Y)**2)
        print('Epoch: ', epoch, ' Accuracy: ', Accuracy)
    
    
X_Test = np.random.randint(2, size=[50,2])
Z_Test = np.array([X_Test[:,0] ^ X_Test[:, 1]]).T
Y_Test, H = Forward(X_Test, W1, B1, W2, B2)
Accuracy = 1 - np.mean((Z_Test - Y_Test)**2)
# =============================================================================
print('Testing Accuracy: ', Accuracy)
# =============================================================================
