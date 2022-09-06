# IMPORTING NECESSARY PACKAGES
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
print(sklearn.__version__)
import matplotlib
print(matplotlib.__version__)

# SET GLOBAL VARIABLES
ModelTrainSize = 0.8
ModelTestSize = 1 - ModelTrainSize

IRIS = sklearn.datasets.load_iris()
IRIS_DATA = IRIS.data
IRIS_TARGET = IRIS.target

print("Iris Dataset (Data): \n", IRIS_DATA)
print("Iris Dataset (Target): \n", IRIS_TARGET)

TrainData, TestData, TrainTarget, TestTarget = train_test_split(IRIS_DATA, IRIS_TARGET, test_size=ModelTestSize, train_size=ModelTrainSize)
print("Iris Dataset (Data) (Train)\n", TrainData)
print("Iris Dataset (Data) (Test)\n", TestData)
print("Iris Dataset (Target) (Train)\n", TrainTarget)
print("Iris Dataset (Target) (Test)\n", TestTarget)

Model = KNeighborsClassifier()
Model.fit(TrainData, TrainTarget)
print("Train Finished, performing test!")
print("Following is the prediction of Test Datas")
print(Model.predict(TestData))
print("Following is the actual Test Dataset's Target")
print(TestTarget)
