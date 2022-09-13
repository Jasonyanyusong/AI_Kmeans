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
gauss_x1 = np.random.normal(loc=0.0, scale=1.0, size=100)
gauss_y1 = np.random.normal(loc=0.0, scale=1.0, size=100)
plt.scatter(gauss_x1, gauss_y1, color='blue')
gauss_x2 = np.random.normal(loc=5.0, scale=1.0, size=100)
gauss_y2 = np.random.normal(loc=5.0, scale=1.0, size=100)
plt.scatter(gauss_x2, gauss_y2, color='red')
gauss_x3 = np.random.normal(loc=10.0, scale=1.0, size=100)
gauss_y3 = np.random.normal(loc=10.0, scale=1.0, size=100)
plt.scatter(gauss_x3, gauss_y3, color='green')
gauss_x4 = np.random.normal(loc=0.0, scale=1.0, size=100)
gauss_y4 = np.random.normal(loc=10.0, scale=1.0, size=100)
plt.scatter(gauss_x4, gauss_y4, color='orange')
gauss_x5 = np.random.normal(loc=10.0, scale=1.0, size=100)
gauss_y5 = np.random.normal(loc=0.0, scale=1.0, size=100)
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

inertia_list = []
calinsk_harabasz_index_list = []
davies_bouldin_score_list = []
silhouette_score_list = []
for i in range(2, len(gauss_xall)):
    gauss_i_predicted = KMeans(n_clusters=i)
    gauss_i_predicted.fit_predict(gauss_all)
    print("********** ", i, " CLUSTER(S) **********")
    print(gauss_i_predicted)
    inertia_list.append(gauss_i_predicted.inertia_)
    print("Inertia: ", gauss_i_predicted.inertia_)
    calinsk_harabasz_index_list.append(metrics.calinski_harabasz_score(gauss_all, gauss_i_predicted.labels_))
    print("Calinsk Harabasz Index: ", metrics.calinski_harabasz_score(gauss_all, gauss_i_predicted.labels_))
    davies_bouldin_score_list.append(metrics.davies_bouldin_score(gauss_all, labels=gauss_i_predicted.labels_))
    print("Davies Bouldin Score: ", metrics.davies_bouldin_score(gauss_all, labels=gauss_i_predicted.labels_))
    silhouette_score_list.append(metrics.silhouette_score(gauss_all, gauss_i_predicted.labels_, metric='euclidean'))
    print("Silhouette Score: ", metrics.silhouette_score(gauss_all, gauss_i_predicted.labels_, metric='euclidean'))
print("Inertia: ", inertia_list)
print("Inertia Length: ", len(inertia_list))

print("Calinsk Harabasz Index: ", calinsk_harabasz_index_list)
print("Calinsk Harabasz Index Length: ", len(calinsk_harabasz_index_list))

print("Davies Bouldin Score: ", davies_bouldin_score_list)
print("Davies Bouldin Score Length: ", len(davies_bouldin_score_list))

print("Silhouette Score: ", silhouette_score_list)
print("Silhouette Score Length: ", len(silhouette_score_list))

Inertia_Chart = matplotlib.pyplot
Inertia_Chart.title("Inertia Plot")
Inertia_Chart.xlabel("Cluster Number:")
Inertia_Chart.ylabel("Inertia:")
Inertia_Chart.plot(range(2, len(inertia_list) + 2), inertia_list)
Inertia_Chart.show()

Calinsk_Harabasz_Index_Chart = matplotlib.pyplot
Calinsk_Harabasz_Index_Chart.title("Calinsk Harabasz Index Plot")
Calinsk_Harabasz_Index_Chart.xlabel("Cluster Number:")
Calinsk_Harabasz_Index_Chart.ylabel("Calinsk Harabasz Index:")
Calinsk_Harabasz_Index_Chart.plot(range(2, len(calinsk_harabasz_index_list) + 2), calinsk_harabasz_index_list)
Calinsk_Harabasz_Index_Chart.show()

Davies_Bouldin_Score_Chart = matplotlib.pyplot
Davies_Bouldin_Score_Chart.title("Davies Bouldin Score Plot")
Davies_Bouldin_Score_Chart.xlabel("Cluster Number:")
Davies_Bouldin_Score_Chart.ylabel("Davies Bouldin Score:")
Davies_Bouldin_Score_Chart.plot(range(2, len(davies_bouldin_score_list) + 2), davies_bouldin_score_list)
Davies_Bouldin_Score_Chart.show()

Silhouette_Score_Chart = matplotlib.pyplot
Silhouette_Score_Chart.title("Silhouette Score Plot")
Silhouette_Score_Chart.xlabel("Cluster Number:")
Silhouette_Score_Chart.ylabel("Silhouette Score Score:")
Silhouette_Score_Chart.plot(range(2, len(silhouette_score_list) + 2), silhouette_score_list)
Silhouette_Score_Chart.show()

print("CODE ENDS")
