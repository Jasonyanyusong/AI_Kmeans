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

gauss_x6 = np.random.normal(loc=0.0, scale=1.0, size=200)
gauss_y6 = np.random.normal(loc=5.0, scale=1.0, size=200)
plt.scatter(gauss_x6, gauss_y6, color='purple')

gauss_x7 = np.random.normal(loc=5.0, scale=1.0, size=200)
gauss_y7 = np.random.normal(loc=0.0, scale=1.0, size=200)
plt.scatter(gauss_x7, gauss_y7, color='yellow')

gauss_x8 = np.random.normal(loc=5.0, scale=1.0, size=200)
gauss_y8 = np.random.normal(loc=10.0, scale=1.0, size=200)
plt.scatter(gauss_x8, gauss_y8, color='brown')

gauss_x9 = np.random.normal(loc=10.0, scale=1.0, size=200)
gauss_y9 = np.random.normal(loc=5.0, scale=1.0, size=200)
plt.scatter(gauss_x9, gauss_y9, color='black')

plt.show()

gauss_xall = np.append(gauss_x1, gauss_x2)
gauss_xall = np.append(gauss_xall, gauss_x3)
gauss_xall = np.append(gauss_xall, gauss_x4)
gauss_xall = np.append(gauss_xall, gauss_x5)
gauss_xall = np.append(gauss_xall, gauss_x6)
gauss_xall = np.append(gauss_xall, gauss_x7)
gauss_xall = np.append(gauss_xall, gauss_x8)
gauss_xall = np.append(gauss_xall, gauss_x9)

gauss_yall = np.append(gauss_y1, gauss_y2)
gauss_yall = np.append(gauss_yall, gauss_y3)
gauss_yall = np.append(gauss_yall, gauss_y4)
gauss_yall = np.append(gauss_yall, gauss_y5)
gauss_yall = np.append(gauss_yall, gauss_y6)
gauss_yall = np.append(gauss_yall, gauss_y7)
gauss_yall = np.append(gauss_yall, gauss_y8)
gauss_yall = np.append(gauss_yall, gauss_y9)

plt.scatter(gauss_xall, gauss_yall, color='black')
plt.show()