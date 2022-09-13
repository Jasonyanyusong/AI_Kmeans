# IMPORT PACKAGES
import sklearn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics

# PRINT PACKAGES VERSION
print(sklearn.__version__)
print(np.__version__)
print(matplotlib.__version__)
print(pd.__version__)

breast_cancer_dataset = load_breast_cancer()
breast_cancer_data = breast_cancer_dataset['data']
print(breast_cancer_data)
print(breast_cancer_data[0])
print(len(breast_cancer_data[0]))

for i in range(0, len(breast_cancer_data), 1):
    print("********** Breast Cancer Number: ", i, "**********")
    print("Value 01: ", breast_cancer_data[i][0])
    print("Value 02: ", breast_cancer_data[i][1])
    print("Value 03: ", breast_cancer_data[i][2])
    print("Value 04: ", breast_cancer_data[i][3])
    print("Value 05: ", breast_cancer_data[i][4])
    print("Value 06: ", breast_cancer_data[i][5])
    print("Value 07: ", breast_cancer_data[i][6])
    print("Value 08: ", breast_cancer_data[i][7])
    print("Value 09: ", breast_cancer_data[i][8])
    print("Value 10: ", breast_cancer_data[i][9])
    print("Value 11: ", breast_cancer_data[i][10])
    print("Value 12: ", breast_cancer_data[i][11])
    print("Value 13: ", breast_cancer_data[i][12])
    print("Value 14: ", breast_cancer_data[i][13])
    print("Value 15: ", breast_cancer_data[i][14])
    print("Value 16: ", breast_cancer_data[i][15])
    print("Value 17: ", breast_cancer_data[i][16])
    print("Value 18: ", breast_cancer_data[i][17])
    print("Value 19: ", breast_cancer_data[i][18])
    print("Value 20: ", breast_cancer_data[i][19])
    print("Value 21: ", breast_cancer_data[i][20])
    print("Value 22: ", breast_cancer_data[i][21])
    print("Value 23: ", breast_cancer_data[i][22])
    print("Value 24: ", breast_cancer_data[i][23])
    print("Value 25: ", breast_cancer_data[i][24])
    print("Value 26: ", breast_cancer_data[i][25])
    print("Value 27: ", breast_cancer_data[i][26])
    print("Value 28: ", breast_cancer_data[i][27])
    print("Value 29: ", breast_cancer_data[i][28])
    print("Value 30: ", breast_cancer_data[i][29])
    print("")

inertia_list = []
calinsk_harabasz_index_list = []
davies_bouldin_score_list = []
silhouette_score_list = []
for i in range(2, len(breast_cancer_data)):
    breast_cancer_i_predicted = KMeans(n_clusters=i)
    breast_cancer_i_predicted.fit_predict(breast_cancer_data)
    print("********** ", i, " CLUSTER(S) **********")
    print(breast_cancer_i_predicted)
    inertia_list.append(breast_cancer_i_predicted.inertia_)
    print("Inertia: ", breast_cancer_i_predicted.inertia_)
    calinsk_harabasz_index_list.append(metrics.calinski_harabasz_score(breast_cancer_data, breast_cancer_i_predicted.labels_))
    print("Calinsk Harabasz Index: ", metrics.calinski_harabasz_score(breast_cancer_data, breast_cancer_i_predicted.labels_))
    davies_bouldin_score_list.append(metrics.davies_bouldin_score(breast_cancer_data, labels=breast_cancer_i_predicted.labels_))
    print("Davies Bouldin Score: ", metrics.davies_bouldin_score(breast_cancer_data, labels=breast_cancer_i_predicted.labels_))
    silhouette_score_list.append(metrics.silhouette_score(breast_cancer_data, breast_cancer_i_predicted.labels_, metric='euclidean'))
    print("Silhouette Score: ", metrics.silhouette_score(breast_cancer_data, breast_cancer_i_predicted.labels_, metric='euclidean'))
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
