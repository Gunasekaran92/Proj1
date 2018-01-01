# Proj1

IRIS DATASET

from sklearn.datasets import load_iris
iris_dataset=load_iris()
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

Keys of iris_dataset: 
dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])

print(iris_dataset['DESCR'][:193] + "\n...")

Iris Plants Database
====================

Notes
-----
Data Set Characteristics:
    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive att
...

print("Target names: {}".format(iris_dataset['target_names']))

Target names: ['setosa' 'versicolor' 'virginica']

print("Feature names: \n{}".format(iris_dataset['feature_names']))

Feature names: 
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

print("Type of Data : {}".format(type(iris_dataset['data'])))

Type of Data : <class 'numpy.ndarray'>

print("Shape of data {}".format(iris_dataset['data'].shape))

Shape of data (150, 4)

print("Print first 5 columns of data :\n{}".format(iris_dataset['data'][:5]))

Print first 5 columns of data :
[[ 5.1  3.5  1.4  0.2]
 [ 4.9  3.   1.4  0.2]
 [ 4.7  3.2  1.3  0.2]
 [ 4.6  3.1  1.5  0.2]
 [ 5.   3.6  1.4  0.2]]

print("Target:\n{}".format(iris_dataset['target']))

Target:
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

print("x_train shape: {}".format(x_train.shape))
print("y_train shape:{}".format(x_test.shape))

x_train shape: (112, 4)
y_train shape:(38, 4)

print("x_test shape: {}".format(x_train.shape))
print("y_test shape:{}".format(x_test.shape))

x_test shape: (112, 4)
y_test shape:(38, 4)

import pandas as pd
import mglearn
import sys
# Create dataframe from data in x_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe=pd.DataFrame(x_train,columns=iris_dataset.feature_names)
# Create a scatter matrix from dataframe,color by x_train
grr=pd.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=0.8,cmap=mglearn.cm3)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train,y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=1, p=2,
           weights='uniform')

import numpy as np
x_new=np.array([[5, 2.9, 1, 0.2]])
print("x_new.shape: {}".format(x_new.shape))

x_new.shape: (1, 4)

prediction=knn.predict(x_new)
print("Prediction: {}".format(prediction))
print("Prediced target name: {}".format(iris_dataset['target_names'][prediction]))

Prediction: [0]
Prediced target name: ['setosa']

y_pred=knn.predict(x_test)
print("Test set prediction:\n {}".format(y_pred))

Test set prediction:
 [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
 2]

print("Test set score: {:.2f}".format(np.mean(y_pred==y_test)))

Test set score: 0.97

print("Test set score: {:.2f}".format(knn.score(x_test,y_test)))

Test set score: 0.97

