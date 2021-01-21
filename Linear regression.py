# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 11:24:54 2020

@author: Lyn
"""


import numpy as np
import pandas as pd 

from matplotlib import pyplot as plt

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression


FILENAME = "../Financial Sample.xlsx" 

#data = pd.read_csv(f+'data_train.txt',delimiter=' ',header=None)

# Product, Unis_sold, Sales_price, Gross Sales, Discounts, Sales, Profit， Data
data_frame = pd.read_excel('C:\\Users\\admin\\Desktop\\Power BI data\\Financial Sample.xlsx', header = 0, usecols = "C, E, G, H ,I,J, L, N, P")
DataArray = data_frame.values  

#print(data_frame.values)
#print(DataArray)

#Unit_sold = DataArray[:, 2]
#print(Unit_sold)
#Price = DataArray][:, 1]


product1_sale = 0

#X = np.zeros(len(DataArray))
#Y = np.zeros(len(DataArray))

X = [] # X is unit sold
Y = [] # Y is Gross 

M = [] # X is unit sold
N = [] # Y is Gross  

for i in DataArray:
    if 'Carretera' in i:
        if 2014 in i:
            X.append(i[4])
            Y.append(i[5])

for j in DataArray:
    if 'Montana' in j:
        if 2014 in j:
            M.append(j[4])
            N.append(j[5])
        
plt.scatter(X,Y, color = 'blue')
plt.scatter(M,N, color = 'green')
plt.ylabel('Sales')
plt.xlabel('Amont of Discount')
        
lrModel = LinearRegression()
lrModel_2 = LinearRegression()

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size = 0.8)
X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(M,N, train_size = 0.8)


X_train = np.array(X_train).reshape(-1, 1)
X_test = np.array(X_test).reshape(-1, 1)


lrModel.fit(X_train,Y_train)
#lrModel.score(X,Y)

X_train_2 = np.array(X_train_2).reshape(-1, 1)
X_test_2 = np.array(X_test_2).reshape(-1, 1)


lrModel_2.fit(X_train_2,Y_train_2)
#lrModel.score(X,Y)

#Linear model for prediction

alpha = lrModel.intercept_
beta = lrModel.coef_

#alpha + beta*X_train

Y_train_pred = lrModel.predict(X_train)
plt.plot(X_train, Y_train_pred, color = 'red')

Y_train_pred_2 = lrModel_2.predict(X_train_2)
plt.plot(X_train_2, Y_train_pred_2, color = 'black')

plt.legend(loc = 2)
plt.show()



