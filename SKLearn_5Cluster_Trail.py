# IMPORT PACKAGES
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.cluster import KMeans

# PRINT PACKAGES VERSION
print(sklearn.__version__)
print(np.__version__)
print(matplotlib.__version__)

# GENERATE COLORED PLOT
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

# plt.show()

# GENERATE UNCOLORED PLOT
gauss_xall = np.append(gauss_x1, gauss_x2)
gauss_xall = np.append(gauss_xall, gauss_x3)
gauss_xall = np.append(gauss_xall, gauss_x4)
gauss_xall = np.append(gauss_xall, gauss_x5)
gauss_yall = np.append(gauss_y1, gauss_y2)
gauss_yall = np.append(gauss_yall, gauss_y3)
gauss_yall = np.append(gauss_yall, gauss_y4)
gauss_yall = np.append(gauss_yall, gauss_y5)
plt.scatter(gauss_xall, gauss_yall, color='black')

# plt.show()

print("GAUSS_XALL")
print(gauss_xall)

print("")
print("GAUSS_YALL")
print(gauss_yall)

print(len(gauss_xall))
print(len(gauss_yall))

gauss_all = []
for i in range (0, len(gauss_xall), 1):
    # gauss_all += np.array([gauss_xall[i],gauss_yall[i]])
    print(i, gauss_xall[i], gauss_yall[i])
    gauss_all.append([gauss_xall[i], gauss_yall[i]])

print(gauss_all)

gauss_y1C_predicted = KMeans(n_clusters=1).fit_predict(gauss_all)
print(gauss_y1C_predicted)

gauss_y2C_predicted = KMeans(n_clusters=2).fit_predict(gauss_all)
print(gauss_y2C_predicted)

gauss_y3C_predicted = KMeans(n_clusters=3).fit_predict(gauss_all)
print(gauss_y3C_predicted)

gauss_y4C_predicted = KMeans(n_clusters=4).fit_predict(gauss_all)
print(gauss_y4C_predicted)

gauss_y5C_predicted = KMeans(n_clusters=5).fit_predict(gauss_all)
print(gauss_y5C_predicted)

gauss_y6C_predicted = KMeans(n_clusters=6).fit_predict(gauss_all)
print(gauss_y6C_predicted)

gauss_y7C_predicted = KMeans(n_clusters=7).fit_predict(gauss_all)
print(gauss_y7C_predicted)

gauss_y8C_predicted = KMeans(n_clusters=8).fit_predict(gauss_all)
print(gauss_y8C_predicted)