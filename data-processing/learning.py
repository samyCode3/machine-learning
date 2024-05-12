import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer



#processing dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
missing_values = dataset.isnull().sum()
#replacing missing dataset
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

num_col = dataset.select_dtypes(include='number')
imputer.fit(num_col)

update = imputer.transform(num_col)
dataset[num_col.columns] = update
print(dataset)
#encodinng dataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],  remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)
#Data preprocessing
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)
