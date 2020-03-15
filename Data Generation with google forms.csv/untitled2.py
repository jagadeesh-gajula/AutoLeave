# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 20:49:50 2020

@author: VickyViper
"""

import pandas as pd

df = pd.read_csv('Data.csv')

from sklearn.linear_model import LogisticRegression


tree = LogisticRegression()
y = df[['Status']]
df = df.drop(["Status","Unnamed: 13",
             "Unnamed: 14",'Timestamp',
             'Company Identity number','Time of Applying for leave',
             'Other thoughts or comments',
             ],axis=1)


df = pd.get_dummies(df)
y = pd.get_dummies(y)['Status_Approved']
df = df.fillna(0)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(df,y)

tree.fit(x_train,y_train)

y_pred = tree.predict(x_test)



x = [y_test - y_pred]

print(x[0])
