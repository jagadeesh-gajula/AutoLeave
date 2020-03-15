#!/usr/bin/env python
# coding: utf-8

# In[573]:


import pandas as pd
df = pd.read_csv('Data.csv')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
tree = LogisticRegression()
y = df['Status']
df = df.drop(["Status","Unnamed: 13",
             "Unnamed: 14",'Timestamp',
             'Company Identity number','Time of Applying for leave',
             'Other thoughts or comments','Same day leave with permission, Select hours',
              'Day of applying for leave','Name'
             ],axis=1)
encoder =  LabelEncoder()
dataframe = df.copy
df.columns


# In[574]:


def enc(x):
    return encoder.fit_transform(x)

for i in df.columns:
    df[i] = enc(df[i].astype(str))


y = enc(y.astype(str))


# In[575]:


log = LogisticRegression()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df,y)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import PySimpleGUI as sg

log = RandomForestClassifier()


log.fit(x_train,y_train)
y_pred = log.predict(x_test)


# In[576]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:





# In[577]:


"""
# All the stuff inside your window.
layout = [  [sg.Text('Enter details')],
          
            [sg.Text('Name'), sg.InputText()],
            [sg.Text('Period of leave'), sg.InputText()],
          
          
            [sg.Text('Place you are applying from')],
            [sg.Radio('Home', "RADIO1",default=True),
             sg.Radio('workplace', "RADIO1"),
            sg.Radio('Out of town', "RADIO1")], 
          
          [sg.Text('Which department do you belong ?')],
            [sg.Radio('cleaning', "RADIO2",default=True),
                sg.Radio('HR', "RADIO2"),
             sg.Radio('maintainace', "RADIO2"),
                sg.Radio('IT', "RADIO2"),
                sg.Radio('marketing', "RADIO2"),
                sg.Radio('Medical', "RADIO2"),
             sg.Radio('Emergency Medical', "RADIO2"),
            sg.Radio('R&D', "RADIO2"),
          sg.Radio('Finance', "RADIO2"),
          sg.Radio('Operations', "RADIO2")],
          
          
        [sg.Text('What is your designation ?')],
            [sg.Radio('Ad-Hoc workers', "RADIO3",default=True),
             sg.Radio('employee', "RADIO3"),
                sg.Radio('foreman', "RADIO3"),
                sg.Radio('manager', "RADIO3"),
                sg.Radio('supervisor', "RADIO3")],
          
          
        [sg.Text('Type of leave ?')],
            [sg.Radio('casual', "RADIO4",default=True),
             sg.Radio('earned', "RADIO4"),
                sg.Radio('Extra ordinary leave', "RADIO4"),
                sg.Radio('Official', "RADIO4"),
                sg.Radio('medical', "RADIO4")],
        
          
          
        [sg.Text('Reason for leave ?')],
            [sg.Radio('death', "RADIO5",default=True),
             sg.Radio('marriage', "RADIO5"),
                sg.Radio('medical', "RADIO5"),
                sg.Radio('Emergency', "RADIO5"),
                sg.Radio('accompany child', "RADIO5"),
            sg.Radio('Duty leave', "RADIO5"),
            sg.Radio('personal work', "RADIO5"),
            sg.Radio('family members in hospital', "RADIO5")],
          
[sg.Submit('Ok'), sg.Button('Cancel')]]


window = sg.Window('Window Title', layout)

while True:
    event, values = window.read()
    if event in (None, 'Cancel'):   
        break
    if event == 'Ok':
        x = values
        break

window.close()
for i in values.keys():
    if values[i]== True:
        values[i]=1
    if values[i]== False:
        values[i]=0
        
if log.predict([values])== 0:
    sg.Popup(store[0],"Sorry, Your leave request can't be approved, Please meet your manager")
if log.predict([values])== 1:
    sg.Popup(store[0],"congrats..! You are granted with leave and enjoy your time")
"""


# In[578]:


"""
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus, graphviz

# If you're on windows:
# Specifing path for dot file.
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'
features=x_train.columns

# plotting tree with max_depth=3
dot_data = StringIO()  
export_graphviz(log, out_file=dot_data,
                feature_names=features, filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_pdf("baby model.pdf")
"""

