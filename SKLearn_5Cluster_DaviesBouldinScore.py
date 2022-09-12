# IMPORT PACKAGES
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn import metrics

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

plt.show()

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

plt.show()

print("GAUSS_XALL")
print(gauss_xall)

print("")
print("GAUSS_YALL")
print(gauss_yall)

print(len(gauss_xall))
print(len(gauss_yall))

gauss_all = []
for i in range(0, len(gauss_xall), 1):
    print(i, gauss_xall[i], gauss_yall[i])
    gauss_all.append([gauss_xall[i], gauss_yall[i]])
print(gauss_all)

davies_bouldin_score_list = []
for i in range(1, len(gauss_xall) + 1):
    gauss_i_predicted = KMeans(n_clusters=i)
    gauss_i_predicted.fit_predict(gauss_all)
    print("********** ", i, " CLUSTER(S) **********")
    print(gauss_i_predicted)
    davies_bouldin_score_list.append(metrics.davies_bouldin_score(gauss_all, labels=gauss_i_predicted.labels_))
    print("Davies Bouldin Score: ", metrics.davies_bouldin_score(gauss_all, labels=gauss_i_predicted.labels_))
print("Davies Bouldin Score: ", davies_bouldin_score_list)

WCSS_Chart = matplotlib.pyplot
WCSS_Chart.title("Davies Bouldin Score Plot")
WCSS_Chart.xlabel("Cluster Number:")
WCSS_Chart.ylabel("Davies Bouldin Score:")
WCSS_Chart.plot(range(1, len(davies_bouldin_score_list) + 1), davies_bouldin_score_list)
WCSS_Chart.show()

print("CODE ENDS")
