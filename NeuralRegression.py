# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 19:44:32 2021

@author: Zikantika
"""

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


#cereal_df = pd.read_csv("/tmp/tmp07wuam09/data/cereal.csv")
diabetic_df2 = pd.read_csv("diabetes.csv")


#print(diabetic_df2.head(5))

#print(diabetic_df2.head(10))

#print(diabetic_df2['Glucose'][0])


#print(diabetic_df2['Glucose'])

X=diabetic_df2['Glucose']
Y=diabetic_df2['BloodPressure']

x, y = np.array(X), np.array(Y)

#x=np.array(x).reshape(1,-1)
##
#y=np.array(y).reshape(1,-1)

x=np.array(x).reshape(-1,1)
#
y=np.array(y).reshape(-1,1)

print(x.shape)
print(y.shape)
#X = [[0., 0.], [1., 1.]]
#y = [[0, 1], [1, 1]]


NeuralNetwork = MLPRegressor(verbose = True,
                             max_iter = 10000,
                             tol = 0,
                             activation = "logistic")
NeuralNetwork.fit(x,y)
#
#NeuralNetwork.fit(X,Y)

print("x shape is ")
print(x.shape)
print("y shape is")
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                     random_state=1)

print(X_train.shape)

X_=np.array(X_test).reshape(-1,1)
print(X_.shape)

Q=NeuralNetwork.predict(X_)

print(Q)

#

Y_=np.array(y_test).reshape(-1,1)

NeuralNetwork.score(X_, Y_)

#Q=NeuralNetwork.predict(10)
#print("Predicted Value is ")
#print(Q)

#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                     hidden_layer_sizes=(1,), random_state=1)
#
#
#clf.fit(x, y)
#
#MLPClassifier(alpha=1e-05, hidden_layer_sizes=(1,), random_state=1,
#              solver='lbfgs')
#
#clf.predict([[1., 2.]])
#
#clf.predict([[0., 0.]])
