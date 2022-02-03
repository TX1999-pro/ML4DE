import numpy as np

def sigm(X, W, B):
    ''' sigmoid function / activation function
        X - input
        W - weight
        B - bias'''
    M = 1/(1+np.exp(-(X.dot(W.T)+B)))
    return M

def diff_W(X, Z, Y, B, W):
    ''' partial differentiation with respect to W
        X - input
        Y - correct output
        Z - current output
        W,B - weight and bias term'''
    dS = sigm(X,W,B)*(1-sigm(X,W,B))
    dW = (Y-Z)*dS

    return X.T.dot(dW)

def diff_B(X, Z, Y, B, W):

    dS = sigm(X,W,B)*(1-sigm(X,W,B))
    dB = (Y-Z)*dS
    print(dB.shape)
    print(dB.sum(axis=0).shape)
    return dB.sum(axis=0)