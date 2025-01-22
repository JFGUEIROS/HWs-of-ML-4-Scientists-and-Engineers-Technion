import numpy as np
import matplotlib.pyplot as plt
# Getting data
x = np.array([1 , 2, 3, 4, 5, 6])
y = np.array([1 , 3, 2, 5, 4, 6])
data = np.concatenate((x[np.newaxis, :], y[np.newaxis, :]), axis=0)
## Calculating stuff
x_avg = np.mean(x)
y_avg = np.mean(y)
beta_1_num = np.sum((x - x_avg) * (y - y_avg))
beta_1_den = np.sum((x - x_avg) ** 2)
beta_1 = beta_1_num / beta_1_den
beta_0 = y_avg - beta_1*x_avg
x_line = np.linspace(1,6,100)

y_line = beta_1*x_line + beta_0
plt.plot(x_line,y_line,"-r",label = "Best line")
plt.plot()
plt.scatter(x, y, color='blue', label='Points')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear regression for blue points')
plt.show()