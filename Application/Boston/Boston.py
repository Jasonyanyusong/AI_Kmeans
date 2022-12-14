# IMPORT PACKAGES
import sklearn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics

# PRINT PACKAGES VERSION
print(sklearn.__version__)
print(np.__version__)
print(matplotlib.__version__)
print(pd.__version__)

boston_data = load_boston()['data']
print(boston_data)

# BASED ON REGRESSION