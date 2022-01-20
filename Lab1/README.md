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

2. How would the learning rate and the number of iteration affect the results? How can we learn it faster and reduce the error further?

3. Does the error go to zero? 
*No.*

 - Why not?

4. How can we make it go exactly zero?
*get the derivative, equate to zero and solve for parameters using the equation, then check if it is indeed a minimum*

5. Can we reproduce XOR in the same procedure?

 - why?

6. How can we solve the XOR by just using one (hidden layer of) neuron?
