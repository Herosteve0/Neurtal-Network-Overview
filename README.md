# Neurtal-Network-Overview
An open source project I made to both understand and share what I've learned about Neural Networks and deep learning as a whole. Coming with different types of code, loss graphs and an analysis for the code complexity (both time and space) along with the math, for anyone who isn't interested in performance and computer science itself.

A few words...

This current implimentation is not well structured for usage, but most of what you need to know is inside of the Program.cs file. If you see this repository and know me in real life, ask me whatever you need and hopefully I'll remember to answer. If not... thanks for finding me?

- The SIMD implementations are the best I could think of and I honestly won't be bothered to increase it anymore.
- As for the GPU, there are issues, for example the loss, max index and softmax bottlenecks the whole training process, making it borderline the same as SIMD. I will probably release a new version to fix those, but I have other projects in mind for the time being.

# Math

## Variables

### $L =$ Layer Length, $\hat{L} =$ Last Layer Length
### $ReLU(x) = \begin{cases} x, x > 0 \\ 0, x \leq 0 \end{cases}$


### Weights: $W^{[L]}_{L×L-1}$
### Bias: $B^{[L]}_L$
### Values: $z^{[L]}_{L}$
### Activations: $a^{[L]}_{L}$
---
### Prediction: $\hat{y}_{\hat{L}}$
### Target: $y_{\hat{L}}$
### Loss: ${C_{0}}_{\hat{L}}$
---
### Delta: $\delta^{[L]}_{L}$

## Layer Linear/Forward

### $z^{[L]} = W^{[L]} × a^{[L-1]} + b^{[L]}$, where $a^{[0]} =$ input vector.

## Layer Activation

### $a^{[L]} = ReLU(z^{[L]})$
### $a^{[\hat{L}]} = Softmax(z^{[\hat{L}]})$

## Loss Function

### $C_0 = -\sum_{k=1}^{\hat{L}} y_{k} \log(\hat{y}_{k})$, since we only have 0 and 1 here, it simplifies to $C_0 = -\log(\hat{y}_{maxIndex{(y)}})$

## Back Propagation

### $\delta^{[\hat{L}]} = a^{[\hat{L}]} - \hat{y}$, because the last layer uses softmax

### $\delta^{[L]} = \frac{\partial C_0}{\partial W^{[L]}} = \frac{\partial z^{[L]}}{\partial W^{[L]}} \frac{\partial a^{[L]}}{\partial z^{[L]}} \frac{\partial C_0}{\partial a^{[L]}} = \begin{cases} (W^{[L+1]})^T \delta^{[L+1]} \odot a^{[L-1]}, z^{[L]} > 0 \\ 0, z^{[L]} \le 0 \end{cases}$

- ### $\frac{\partial z^{[L]}}{\partial W^{[L]}} = a^{[L-1]}$
- ### $\frac{\partial a^{[L]}}{\partial z^{[L]}} = ReLU'(z^{[L]}) = \begin{cases} 1, z^{[L]} > 0 \\ 0, z^{[L]} \leq 0 \end{cases}$
- ### $\frac{\partial C_0}{\partial a^{[L]}} = (W^{[L+1]})^T \delta^{[L+1]}$

### Same idea for $b^{[L]}$ and $a^{[L-1]}$