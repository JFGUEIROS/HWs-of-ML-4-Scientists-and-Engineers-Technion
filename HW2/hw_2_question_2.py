import numpy as np
import matplotlib.pyplot as plt
x = np.array([1 , 2, 3, 4, 5, 6])
y = np.array([1 , 3, 2, 5, 4, 6])

def cost_function(beta_0,beta_1,x,y):
    h = beta_0*1 + beta_1*x
    J = 1/(2*len(x)) * np.sum((h - y)**2)

    return J

def gradient_descent(alpha,iter,beta_0,beta_1,x,y,x0):
    cost = np.zeros(iter)
    for i in range(iter):
        h = beta_0*1 + beta_1*x
        gradient_0 = 1/(len(x)) * np.sum(h-y) * x0
        beta_0 = beta_0 - alpha*gradient_0 
        gradient_1 = 1/(len(x)) * np.sum((h-y)*x)
        beta_1 = beta_1 - alpha*gradient_1
        cost[i] = cost_function(beta_0,beta_1,x,y)

    return [cost,beta_0,beta_1]

alpha = 0.01
iter = 1000
lins = np.linspace(1,iter,iter)
x0 = 1
beta_0 = 1
beta_1 = 1
[cost,beta_0,beta_1] = gradient_descent(alpha,iter,beta_0,beta_1,x,y,x0)
x_line = np.linspace(1,6,100)
y_line = beta_1*x_line + beta_0

# Figure 1: Plot of cost
plt.figure(1)  # Start a new figure
plt.plot(lins, cost, label="Cost")
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Convergence')
plt.legend()
plt.show()

# Figure 2: Linear regression fit
plt.figure(2)  # Start a new figure
plt.plot(x_line, y_line, "-r", label="Best Line")  # Best-fit line
plt.scatter(x, y, color="blue", label="Data Points")  # Data points
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression for Blue Points")
plt.legend()
plt.show()
