import numpy as np
import matplotlib.pyplot as plt

examples = np.array([[[1, 1, 1, -1], 
                     [-1, 1, -1, -1],
                     [-1, 1, -1, -1],
                     [-1, 1, -1, -1]],
                    [[-1, 1, 1, 1],
                     [-1, -1, 1, -1],
                     [-1, -1, 1, -1],
                     [-1, -1, 1, -1]],
                    [[-1, -1, 1, -1],
                     [-1, -1, 1, -1],
                     [1, -1, 1, -1],
                     [1, 1, 1, -1]],
                    [[-1, -1, -1, 1],
                     [-1, -1, -1, 1],
                     [-1, 1, -1, 1],
                     [-1, 1, 1, 1]]])

fig = plt.figure(0, (6, 6))
for i in range(len(examples)):
    fig.add_subplot(2, 2, i+1)
    plt.imshow(examples[i])

y = np.array([1, 1, -1, -1])

X = np.hstack((examples.reshape(-1, 16), np.ones((len(y), 1))))

w = np.zeros(17)
lr = 1.0

i = 1
yhat = np.dot(X[i], w)

w += lr*X[i]
i += 1

yhat = np.dot(X[i], w)
w -= lr*X[i]

i += 1

yhat = np.dot(X[i], w)
w -= lr*X[i]
i = 0


yhat = np.dot(X[i], w)

for i in range(4):
    yhat = np.dot(X[i], w)
    print(yhat, y[i])
