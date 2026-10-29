import numpy as np

a = [0, 1, 2]
b = [2, 2, 2]

def neuron(x, w):
  return np.dot(x, w) > 0
print(neuron(a,b))