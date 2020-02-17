# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Dataset 'Data.csv' and this script to be located in project root map
PROJECT_ROOT = r'C:\Users\HelenevanEttinger-Ve\Documents\Tutorials\P14-Data-Preprocessing\Data_Preprocessing'
os.chdir(PROJECT_ROOT)

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Print a preview of X and y with a maximum of 10 lines
max_num = min(10, y.size)
print('X = ', X[0: max_num], '\n', 'y = ', y[0: max_num])

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Encoding categorical data by using dummy variables
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder="passthrough")
X = np.array(ct.fit_transform(X), dtype=np.float)

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Preview of data after preprocessing
print('X = ', X[0: max_num], '\n', 'y = ', y[0: max_num])
