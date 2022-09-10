from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

mean = np.array([2.5, 3.5])
cov = np.array([[0.5, 0.2], [0.2, 0.6]])
n = 1000
v = np.linspace(0, 6, n)

gauss_x = norm.pdf(v, loc=mean[0], scale=cov[0, 0]**0.5)
print("GAUSS X", gauss_x)
gauss_y = norm.pdf(v, loc=mean[1], scale=cov[1, 1]**0.5)
print("")
print("GAUSS Y", gauss_y)

plt.scatter(gauss_x,gauss_y,marker='.',color='red')
#plt.scatter([1,2,3],[1,2,3])
plt.show()