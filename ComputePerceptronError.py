import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
y = np.array([[-1], [1], [1], [1]])
w0_range = np.arange(-1.9, 2.0, 0.2)
w1_range = np.arange(-1.9, 2.0, 0.2)
b = 1

w0_points = []
w1_points = []
error_points = []

for w0 in w0_range:
    for w1 in w1_range:
        yhat = X[:,0]*w0 + X[:,1]*w1 + b
        error = np.mean((y.ravel() - yhat)**2)

        w0_points.append(w0)
        w1_points.append(w1)
        error_points.append(error)

w0_points = np.array(w0_points)
w1_points = np.array(w1_points)
error_points = np.array(error_points)


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')


scatter = ax.scatter(w0_points, w1_points, error_points,
c=error_points, cmap='viridis',
alpha=0.6, s=20)

ax.set_xlabel('Weight w0')
ax.set_ylabel('Weight w1')
ax.set_zlabel('Mean Squared Error')
ax.set_title('Perceptron Error Surface\n(Linearly Separable Case)')


plt.show()