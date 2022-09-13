import sklearn.datasets as dts
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

wine_dataset = dts.load_wine()
wine_dataset_data = wine_dataset.data
print(len(wine_dataset_data))
print(wine_dataset_data)

for i in range(0, len(wine_dataset_data), 1):
    print("********** Wine Number: ", i, "**********")
    print("Value 01: ", wine_dataset_data[i][0])
    print("Value 02: ", wine_dataset_data[i][1])
    print("Value 03: ", wine_dataset_data[i][2])
    print("Value 04: ", wine_dataset_data[i][3])
    print("Value 05: ", wine_dataset_data[i][4])
    print("Value 06: ", wine_dataset_data[i][5])
    print("Value 07: ", wine_dataset_data[i][6])
    print("Value 08: ", wine_dataset_data[i][7])
    print("Value 09: ", wine_dataset_data[i][8])
    print("Value 10: ", wine_dataset_data[i][9])
    print("Value 11: ", wine_dataset_data[i][10])
    print("Value 12: ", wine_dataset_data[i][11])
    print("Value 13: ", wine_dataset_data[i][12])
    print("")

inertia_list = []
calinsk_harabasz_index_list = []
davies_bouldin_score_list = []
silhouette_score_list = []
for i in range(2, len(wine_dataset_data)):
    wine_i_predicted = KMeans(n_clusters=i)
    wine_i_predicted.fit_predict(wine_dataset_data)
    print("********** ", i, " CLUSTER(S) **********")
    print(wine_i_predicted)
    inertia_list.append(wine_i_predicted.inertia_)
    print("Inertia: ", wine_i_predicted.inertia_)
    calinsk_harabasz_index_list.append(metrics.calinski_harabasz_score(wine_dataset_data, wine_i_predicted.labels_))
    print("Calinsk Harabasz Index: ", metrics.calinski_harabasz_score(wine_dataset_data, wine_i_predicted.labels_))
    davies_bouldin_score_list.append(metrics.davies_bouldin_score(wine_dataset_data, labels=wine_i_predicted.labels_))
    print("Davies Bouldin Score: ", metrics.davies_bouldin_score(wine_dataset_data, labels=wine_i_predicted.labels_))
    silhouette_score_list.append(metrics.silhouette_score(wine_dataset_data, wine_i_predicted.labels_, metric='euclidean'))
    print("Silhouette Score: ", metrics.silhouette_score(wine_dataset_data, wine_i_predicted.labels_, metric='euclidean'))
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
