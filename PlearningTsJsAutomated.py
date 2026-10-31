import numpy as np
import matplotlib.pyplot as plt

examples=np.array([[[1,1,1,-1], #T Shifted Left
[-1,1,-1,-1],
[-1,1,-1,-1],
[-1,1,-1,-1]],
[[-1,1,1,1], #T Shifted Right
[-1,-1,1,-1],
[-1,-1,1,-1],
[-1,-1,1,-1]],
[[-1,-1,1,-1], #J Shifted Left
[-1,-1,1,-1],
[1,-1,1,-1],
[1,1,1,-1]],
[[-1,-1,-1,1], #J Shifted Right
[-1,-1,-1,1],
[-1,1,-1,1],
[-1,1,1,1]]])

fig=plt.figure(0,(6,6))
for i in range(len(examples)):
    fig.add_subplot(2,2,i+1)
    plt.imshow(examples[i])
plt.show()

y = np.array([1, 1, -1, -1])

X = np.hstack((examples.reshape(-1, 16), np.ones((len(y), 1))))

w = np.zeros(17)
lr = 1.0

for i in range(1, 8):
    yhat = np.dot(X[i%len(y)], w)
    if yhat <= 0 and y[i%len(y)]>0:
        print(f"output is {yhat} but we want it to be {y[i%len(y)]}, updating weights.")
        w += lr*X[i%len(y)]
    elif yhat > 0 and y[i%len(y)] <= 0:
        print(f"output is {yhat} but we want it to be {y[i%len(y)]}, updating weights.")
        w -= lr*X[i%len(y)]
    else:
        print(f"output is {yhat}, which has the same sign as our target {y[i%len(y)]},"
              f"machine is correct, not updating weights.")