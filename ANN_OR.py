from ANN_func import sigm, diff_W, diff_B
import numpy as np

#initialise default
W= np.random.randn(1,2) # weight
B = np.random.randn(1) # bias

X = np.random.randint(2, size=[15,2])
Y = np.array([X[:,0] | X[:,1]]).T # WHY? 
# this takes the OR value for input X

X_Test = np.random.randint(2, size=[15,2])
Y_Test = np.array([X_Test[:,0] | X_Test[:,1]]).T

learning_rate = 0.01

for epoch in range(5000):

    output = sigm(X,W,B)

    W += learning_rate * diff_W(X, output, Y, B, W).T
    B += learning_rate * diff_B(X, output, Y, B, W)
    if not epoch % 50:
        Accuracy = 1 - np.mean((output-Y)**2)
        # Accuracy = 1 - np.mean(abs(output-Y))
        print('Epoch: ', epoch, 'Accuracy: ', Accuracy, 'Weight', W, 'B:', B)

# Check accuracy using the test set
output_Test = sigm(X_Test,W,B)
Accuracy_Test = 1 - np.mean((output_Test-Y_Test)**2)

print('Accuary for test set: ', Accuracy)