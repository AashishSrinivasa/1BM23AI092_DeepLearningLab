#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import matplotlib.pyplot as plt
import numpy as np
class Perceptron:
    def __init__(self, weight=1.0, bias=0.0, activation='linear'):
        self.weight = weight
        self.bias = bias
        self.activation = activation.lower()

    def activate(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Unsupported activation function. Choose 'linear', 'sigmoid', or 'tanh'.")

    def predict(self, input_values):
        z = self.weight * input_values + self.bias
        return self.activate(z)
def plot_perceptron_outputs(weight=1.0, bias=0.0):
    input_range = np.linspace(-10, 10, 500)

    activations = ['linear', 'sigmoid', 'tanh']
    plt.figure(figsize=(10, 6))

    for activation in activations:
        p = Perceptron(weight=weight, bias=bias, activation=activation)
        output = p.predict(input_range)
        plt.plot(input_range, output, label=f'{activation} activation')

    plt.title("Perceptron Output with Different Activation Functions")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.legend()
    plt.show()
if __name__ == "__main__":
    plot_perceptron_outputs(weight=1.0, bias=0.0)


# In[ ]:




