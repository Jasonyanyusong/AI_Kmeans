import numpy as np
import matplotlib.pyplot as plt

gauss_x1 = np.random.normal(loc=0.0, scale=1.0, size=200)
gauss_y1 = np.random.normal(loc=0.0, scale=1.0, size=200)
plt.scatter(gauss_x1, gauss_y1, color='blue')

gauss_x2 = np.random.normal(loc=5.0, scale=1.0, size=200)
gauss_y2 = np.random.normal(loc=5.0, scale=1.0, size=200)
plt.scatter(gauss_x2, gauss_y2, color='red')

gauss_x3 = np.random.normal(loc=10.0, scale=1.0, size=200)
gauss_y3 = np.random.normal(loc=10.0, scale=1.0, size=200)
plt.scatter(gauss_x3, gauss_y3, color='green')

gauss_x4 = np.random.normal(loc=0.0, scale=1.0, size=200)
gauss_y4 = np.random.normal(loc=10.0, scale=1.0, size=200)
plt.scatter(gauss_x4, gauss_y4, color='orange')

gauss_x5 = np.random.normal(loc=10.0, scale=1.0, size=200)
gauss_y5 = np.random.normal(loc=0.0, scale=1.0, size=200)
plt.scatter(gauss_x5, gauss_y5, color='pink')

plt.show()

gauss_xall = np.append(gauss_x1, gauss_x2)
gauss_xall = np.append(gauss_xall, gauss_x3)
gauss_xall = np.append(gauss_xall, gauss_x4)
gauss_xall = np.append(gauss_xall, gauss_x5)

gauss_yall = np.append(gauss_y1, gauss_y2)
gauss_yall = np.append(gauss_yall, gauss_y3)
gauss_yall = np.append(gauss_yall, gauss_y4)
gauss_yall = np.append(gauss_yall, gauss_y5)

plt.scatter(gauss_xall, gauss_yall, color='black')

plt.show()

print("GAUSS_XALL")
print(gauss_xall)

print("")
print("GAUSS_YALL")
print(gauss_yall)