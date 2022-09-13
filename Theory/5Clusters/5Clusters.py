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

gauss_1_predicted = KMeans(n_clusters=1)
gauss_1_predicted.fit_predict(gauss_all)
print("********** ", 1, " CLUSTER(S) **********")
print(gauss_1_predicted)
inertia_list.append(gauss_1_predicted.inertia_)
print("Inertia: ", gauss_1_predicted.inertia_)
print("========================================")
print("")

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
    print("========================================")
    print("")

gauss_500_predicted = KMeans(n_clusters=500)
gauss_500_predicted.fit_predict(gauss_all)
print("********** ", 500, " CLUSTER(S) **********")
print(gauss_500_predicted)
inertia_list.append(gauss_500_predicted.inertia_)
print("Inertia: ", gauss_500_predicted.inertia_)
print("========================================")
print("")

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

print("DISPLAY TWO KEY POINT ON INERTIA PLOT")
print("POINT #001: ", "X = ", 1, "Y = ", inertia_list[0])
print("POINT #150: ", "X = ", len(inertia_list), "Y = ", inertia_list[len(inertia_list) - 1])

print("CALCULATE LINEAR EQUATION OF THE TWO KEY POINT ON INERTIA PLOT")
linear_k = (inertia_list[len(inertia_list) - 1] - inertia_list[0]) / (len(inertia_list) - 1)
linear_b = inertia_list[0] - linear_k * 1
print("LINEAR EQUATION: ", "y = ", linear_k, "x + ", linear_b, "(x != 0)")

print("CALCULATE PLOTS ON THE LINE (AVERAGE CHANGE RATE OF INERTIA)")
linear_inertia_points_list = []
plot_real_inertia_list = []
for i in range(1, len(inertia_list), 1):
    linear_inertia_point_yVal = i * linear_k + linear_b
    linear_inertia_points_list.append(linear_inertia_point_yVal)
    plot_real_inertia_list.append(inertia_list[i - 1])
    print("Point Number: ", i, " X = ", i, " Y = ", linear_inertia_point_yVal)

print("PLOT ENHANCED INERTIA")
Enhanced_Inertia_Plot = matplotlib.pyplot
Enhanced_Inertia_Plot.title("Enhanced Inertia Plot")
Enhanced_Inertia_Plot.xlabel("Cluster Number:")
Enhanced_Inertia_Plot.ylabel("Inertia")
Enhanced_Inertia_Plot.plot(range(1, len(plot_real_inertia_list) + 1), plot_real_inertia_list)
Enhanced_Inertia_Plot.plot(range(1, len(linear_inertia_points_list) + 1), linear_inertia_points_list)
Enhanced_Inertia_Plot.show()

print("CALCULATE AND PLOT RELATIVE GAIN COEFFICIENT")
Relative_Gain_Coefficient_Plot = matplotlib.pyplot
Relative_Gain_Coefficient_Plot.title("Relative Gain Coefficient Plot")
Relative_Gain_Coefficient_Plot.xlabel("Cluster Number:")
Relative_Gain_Coefficient_Plot.ylabel("Relative Gain Coefficient:")
relative_gain_coefficient_list = []
for i in range(1, len(linear_inertia_points_list), 1):
    print("PLIL Length: ", len(plot_real_inertia_list), " Current i: ", i)
    print("Real Inertia: ", plot_real_inertia_list[i - 1])
    print("LINEAR AVERAGE INERTIA: ", linear_inertia_points_list[i])
    Relative_Gain_Coefficient = (linear_inertia_points_list[i] - plot_real_inertia_list[i]) / linear_inertia_points_list[0]
    print("RELATIVE GAIN COEFFICIENT: ", Relative_Gain_Coefficient)
    relative_gain_coefficient_list.append(Relative_Gain_Coefficient)
Relative_Gain_Coefficient_Plot.plot(range(1, len(relative_gain_coefficient_list) + 1), relative_gain_coefficient_list)
Relative_Gain_Coefficient_Plot.show()

print("CODE ENDS")

