import numpy as np

#initialise
W1= np.random.randn(3,2) # weight
B1 = np.random.randn(3) # bias
W2 = np.random.randn(1,3)
B2 = np.random.randn(1)
#.dot - inner product
#.T - transpose


def sigm(X, W, B):
    ''' sigmoid function / activation function
        X - input
        W - weight
        B - bias'''
    M = 1/(1+np.exp(-(X.dot(W.T)+B)))
    return M


def Forward(X, W1, B1, W2, B2):
    # first layer

    H = sigm(X, W1, B1)

    # second layer

    Y = sigm(H, W2, B2)

    # we return both the final output and the output from the hidden layer

    return Y, H
''' partial differentiation with respect to W
    X - input
    Y - correct output
    Z - current output
    W,B - weight and bias term
'''
# the diff are negative, so the gradient descent can work
    
def diff_B2(Z, Y):
    # layer 2 bias
    dB = (Z-Y)*Y*(1-Y)
    return dB.sum(axis=0) # SUM BY ROW
    # why do we take the sum?

def diff_W2(H, Z, Y):
    # layer 2
    dW = (Z-Y)*Y*(1-Y)
    return H.T.dot(dW)

def diff_W1(X, H, Z, Y, W2):
    # layer 1
    dZ = (Z-Y).dot(W2)*Y*(1-Y)*H*(1-H)
    return X.T.dot(dZ)

def diff_B1(Z, Y, W2, H):
    # layer 1
    return((Z-Y).dot(W2)*Y*(1-Y)*H*(1-H)).sum(axis=0)

print("ANN for XOR")
X = np.random.randint(2, size=[15,2])
Z = np.array([X[:,0] != X[:,1]]).T 
# this takes the XOR value for input X

X_Test = np.random.randint(2, size=[15,2])
Y_Test = np.array([X_Test[:,0] != X_Test[:,1]]).T

learning_rate = 0.01

for epoch in range(10000):

    Y, H = Forward(X, W1, B1, W2, B2)

    W2 += learning_rate * diff_W2(H,Z,Y).T
    B2 += learning_rate * diff_B2(Z,Y)
    W1 += learning_rate * diff_W1(X,H,Z,Y,W2).T
    B1 += learning_rate * diff_B1(Z, Y, W2, H)
    if not epoch % 50:
        Accuracy = 1 - np.mean((Z-Y)**2)
        print('Epoch: ', epoch, 'Accuracy: ', Accuracy)
