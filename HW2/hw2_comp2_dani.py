import matplotlib.pyplot as plt
import numpy as np

x_points = np.array([1, 2, 3, 4, 5, 6])
y_points = np.array([1, 3, 2, 4, 5, 6])


def lin_regression(x, y, x0, iterations, alpha):
    m = len(x)
    beta0 = 0
    beta1 = 0
    cost = []

    for i in range(iterations):
        h = beta0 * 1 + beta1 * x
        cost_term = 1 / (2 * m) * np.sum((h - y) ** 2)
        grad0 = (1 / m) * np.sum(h - y) * x0
        grad1 = (1 / m) * np.sum((h - y) * x)

        cost.append(cost_term)

        beta0 -= alpha * grad0
        beta1 -= alpha * grad1
        
    return [cost, beta0, beta1]

data = lin_regression(x_points, y_points, 1, 1000, 0.1)
cost = data[0]
beta0 = data[1]
beta1 = data[2]


print(beta0, beta1)

plt.figure()
plt.plot(range(len(cost)), cost)
plt.title("Cost as a function of the Iteration")
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()
plt.close()

plt.figure()
plt.plot(x_points, y_points, "yo", label="Data Points")
plt.plot(x_points, beta0 + beta1 * x_points, "--k", label="Best Fit Line")
plt.title("Data Points and Best Fit Line")
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
