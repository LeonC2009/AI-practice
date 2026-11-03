import numpy as np
import matplotlib.pyplot as plt

examples = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])

y = np.array([-1, 1, 1, -1])

X = np.hstack((examples.reshape(-1, 2), np.ones((len(y), 1))))

w = np.zeros(3)
lr = 1.0

for i in range(1, 14):
    yhat = np.dot(X[i%len(y)], w)
    print(f"step: {i}, current example: {X[i%len(y)][:2]}, current weights = {w}")
    if yhat <= 0 and y[i%len(y)] > 0:
        print(f"output is {yhat} but we want it to be {y[i%len(y)]}, updating weights.")
        w += lr*X[i%len(y)]
    elif yhat > 0 and y[i%len(y)] <= 0:
        print(f"output is {yhat} but we want it to be {y[i%len(y)]}, updating weights.")
        w -= lr*X[i%len(y)]
    else:
        print(f"output is {yhat}, which has the same sign as our target {y[i%len(y)]},"
        f"machine is correct, not updating weights.")