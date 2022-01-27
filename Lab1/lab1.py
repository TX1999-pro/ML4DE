import numpy as np

#initialise
W= np.random.randn(1,2) # weight
B = np.random.randn(1) # bias

#.dot - inner product
#.T - transpose

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
    return dB.sum(axis=0)

print("ANN for OR")
X = np.random.randint(2, size=[15,2]) 
# randint select a random integer below the first argument
Y = np.array([X[:,0] | X[:,1]]).T # WHY? 
# this takes the OR value for input X

X_Test = np.random.randint(2, size=[15,2])
Y_Test = np.array([X_Test[:,0] | X_Test[:,1]]).T #

learning_rate = 0.01

for epoch in range(500):

    output = sigm(X,W,B)

    W += learning_rate * diff_W(X, output, Y, B, W).T
    B += learning_rate * diff_B(X, output, Y, B, W)
    if not epoch % 50:
        Accuracy = 1 - np.mean((output-Y)**2)
        print('Epoch: ', epoch, 'Accuracy: ', Accuracy)

print("ANN for AND")

X = np.random.randint(2, size=[15,2])
Y = np.array([X[:,0] & X[:,1]]).T # this takes the OR value for input X

X_Test = np.random.randint(2, size=[15,2])
Y_Test = np.array([X_Test[:,0] & X_Test[:,1]]).T

learning_rate = 0.01

for epoch in range(500):

    output = sigm(X,W,B)

    W += learning_rate * diff_W(X, output, Y, B, W).T
    B += learning_rate * diff_B(X, output, Y, B, W)
    if not epoch % 50:
        Accuracy = 1 - np.mean((output-Y)**2)
        print('Epoch: ', epoch, 'Accuracy: ', Accuracy)

'''
NOT SURE IF THE FOLLOWING IS CORRECT
'''
print("ANN for XOR")

X = np.random.randint(2, size=[15,2])
Y = np.array([X[:,0] != X[:,1]]).T # this takes the OR value for input X

X_Test = np.random.randint(2, size=[15,2])
Y_Test = np.array([X_Test[:,0] | X_Test[:,1]]).T

learning_rate = 0.01

for epoch in range(500):

    output = sigm(X,W,B)

    W += learning_rate * diff_W(X, output, Y, B, W).T
    B += learning_rate * diff_B(X, output, Y, B, W)
    if not epoch % 50:
        Accuracy = 1 - np.mean((output-Y)**2)
        print('Epoch: ', epoch, 'Accuracy: ', Accuracy)