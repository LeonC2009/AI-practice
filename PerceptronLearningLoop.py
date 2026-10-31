import numpy as np

a = [0, 1, 2]
b = [2, 2, 2]
label = [1, 1, 1]

def neuron(x, w):
  return np.dot(x, w) > 0
print(neuron(a,b))

def update_neuron(x, w, label):
  is_correct = False
  prediction = neuron(x, w)
  if prediction == 0 and label == 1:
    w += x
  elif prediction == 1 and label == 0:
    w -= x
  else:
    is_correct = True
  return w, is_correct, prediction
print(update_neuron(a, b, label))