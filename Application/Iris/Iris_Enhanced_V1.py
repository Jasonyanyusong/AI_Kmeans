# IMPORT PACKAGES
import sklearn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics

# PRINT PACKAGES VERSION
print(sklearn.__version__)
print(np.__version__)
print(matplotlib.__version__)
print(pd.__version__)

dataset_iris = load_iris()
print(type(dataset_iris))
print(dataset_iris.keys())

# TEST DATASET STRUCTURE
print(dataset_iris)
print("**********")
print(dataset_iris['data'])
# TYPE: - sepal length in cm\n        - sepal width in cm\n        - petal length in cm\n        - petal width in cm\n
print(len(dataset_iris['data']))
print(dataset_iris['data'][0])
print(dataset_iris['data'][len(dataset_iris) - 1])
print(dataset_iris['data'][0][0])

print(len(dataset_iris['target']))
print(dataset_iris['target'])

iris_data = dataset_iris['data']
print(iris_data)

# CHECK IRIS_DATA's DATASTRUCTURE
for i in range(0,len(dataset_iris['data']),1):
    print("********** Iris Number: ", i, "**********")
    print("Sepal Length (cm): ", iris_data[i][0])
    print("Sepal Width (cm): ", iris_data[i][1])
    print("Petal Length (cm): ", iris_data[i][2])
    print("Petal Width (cm): ", iris_data[i][3])
    print("")

inertia_list = []
calinsk_harabasz_index_list = []
davies_bouldin_score_list = []
silhouette_score_list = []

iris_1_predicted = KMeans(n_clusters=1)
iris_1_predicted.fit_predict(iris_data)
print("********** ", 1, " CLUSTER(S) **********")
print(iris_1_predicted)
inertia_list.append(iris_1_predicted.inertia_)
print("Inertia: ", iris_1_predicted.inertia_)
print("========================================")

for i in range(2, len(iris_data)):
    iris_i_predicted = KMeans(n_clusters=i)
    iris_i_predicted.fit_predict(iris_data)
    print("********** ", i, " CLUSTER(S) **********")
    print(iris_i_predicted)
    inertia_list.append(iris_i_predicted.inertia_)
    print("Inertia: ", iris_i_predicted.inertia_)
    calinsk_harabasz_index_list.append(metrics.calinski_harabasz_score(iris_data, iris_i_predicted.labels_))
    print("Calinsk Harabasz Index: ", metrics.calinski_harabasz_score(iris_data, iris_i_predicted.labels_))
    davies_bouldin_score_list.append(metrics.davies_bouldin_score(iris_data, labels=iris_i_predicted.labels_))
    print("Davies Bouldin Score: ", metrics.davies_bouldin_score(iris_data, labels=iris_i_predicted.labels_))
    silhouette_score_list.append(metrics.silhouette_score(iris_data, iris_i_predicted.labels_, metric='euclidean'))
    print("Silhouette Score: ", metrics.silhouette_score(iris_data, iris_i_predicted.labels_, metric='euclidean'))
    print("========================================")

iris_150_predicted = KMeans(n_clusters=150)
iris_150_predicted.fit_predict(iris_data)
print("********** ", 150, " CLUSTER(S) **********")
print(iris_150_predicted)
inertia_list.append(iris_150_predicted.inertia_)
print("Inertia: ", iris_150_predicted.inertia_)
print("========================================")

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
Inertia_Chart.plot(range(1, len(inertia_list) + 1), inertia_list)
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
