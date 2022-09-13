# IMPORT PACKAGES
import sklearn
import numpy as np
import matplotlib
from sklearn.datasets import load_iris
import pandas as pd

# PRINT PACKAGES VERSION
print(sklearn.__version__)
print(np.__version__)
print(matplotlib.__version__)
print(pd.__version__)

dataset_iris = load_iris()
print(type(dataset_iris))
print(dataset_iris.keys())
