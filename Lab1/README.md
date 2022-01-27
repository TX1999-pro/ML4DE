# Laboratory 1 - Neural Networks Basics

## Objective:
1. Understand and code a simple neuron
2. Understand how a neuron learns
3. Understand its limitations

A Neuron. 
Source: https://towardsdatascience.com/statistics-is-freaking-hard-wtf-is-activation-function-df8342cdf292

## Key Concepts:
- sigmoid function
- error function (cost function)
- gradient descend to update
- training and test set for reliability


## Question to ponder:
1. How the weights and error change with each iteration?
- error decreases
- the weight *increases* - but they become **closer to each other** - equal weight?
```
Epoch:  0 Accuracy:  0.7649667585641837 Weight [[-0.27624644  0.55248897]]
Epoch:  50 Accuracy:  0.7815058221912975 Weight [[-0.08530497  0.61380266]]
Epoch:  100 Accuracy:  0.7986580694792352 Weight [[0.09529666 0.67060967]]
Epoch:  150 Accuracy:  0.8162858345779926 Weight [[0.27147419 0.72571588]]
Epoch:  200 Accuracy:  0.8337402990727474 Weight [[0.44540737 0.7804202 ]]
Epoch:  250 Accuracy:  0.8502792809737887 Weight [[0.61696638 0.83511705]]
Epoch:  300 Accuracy:  0.8653520181341084 Weight [[0.784778   0.88968197]]
Epoch:  350 Accuracy:  0.8787005207151874 Weight [[0.94711024 0.94378028]]
Epoch:  400 Accuracy:  0.8903120477181103 Weight [[1.10246992 0.99708087]]
Epoch:  450 Accuracy:  0.9003196263023798 Weight [[1.24987027 1.04935733]]
...

Epoch:  4750 Accuracy:  0.9908717441695604 Weight [[4.14959049 4.15602451]]
Epoch:  4800 Accuracy:  0.99097973559781 Weight [[4.16304082 4.16933007]]
Epoch:  4850 Accuracy:  0.9910853577228027 Weight [[4.17634157 4.18249071]]
Epoch:  4900 Accuracy:  0.991188685017261 Weight [[4.18949589 4.19550938]]
Epoch:  4950 Accuracy:  0.9912897889376061 Weight [[4.20250681 4.20838892]]
Accuary for test set:  0.9912897889376061
```

2. How would the learning rate and the number of iteration affect the results? How can we learn it faster and reduce the error further?
- small learning rate will make the gradient descent very *slow*
- if learning rate is too large, the result might overshoot and be inaccurate

3. Does the error go to zero? 
*No.*

Why not?
- floating error from computer
- theoretically for convex function, gradient descent can converge but in practice it may take infinite iterations

4. How can we make it go exactly zero?

*get the derivative analytically, equate to zero and solve for parameters using the equation, then check if it is indeed a minimum*

5. Can we reproduce XOR in the same procedure?

Yes? - can we assign more input for comparison? say, X1_1, X1_2, X2_1, X2_2 - WRONG
No ? - not enough information
**why?**

6. How can we solve the XOR by just using one (hidden layer of) neuron?

Make AND and  OR as the hidden layer and perform the same gradient descent - can we do that ?? Will the error function be the same?

7. Where does the exponential of 2 disappeared?

It