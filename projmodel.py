# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 18:10:53 2022

@author: noora
"""

# Prediction Model
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
data=pd.read_csv('sensor.csv')
data['date'] = data['timestamp'].apply(lambda x: x.split(' ')[0])
data['time'] = data['timestamp'].apply(lambda x: x.split(' ')[1])
data= data.drop(['timestamp'], axis=1)
data['month'] = pd.to_datetime(data['date']).dt.month
data['sensor_00']=data['sensor_00'].fillna(data['sensor_00'].median())
data['sensor_50']=data['sensor_50'].fillna(data['sensor_50'].median())
data['sensor_51']=data['sensor_51'].fillna(data['sensor_51'].median())
data= data.fillna(0)
from sklearn.preprocessing import LabelEncoder
laben =LabelEncoder()
data['machine_status']=laben.fit_transform(data['machine_status'])
a=['date','time']
for i in np.arange(len(a)):
    data[a[i]]=laben.fit_transform(data[a[i]])
x=data.drop(['machine_status','date','time','month'],axis=1)
y=pd.DataFrame(data['machine_status'])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=32)
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
data = scalar.fit_transform(data)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
#Saving the model to disk
pickle.dump(rf,open('projmodel.pkl','wb') )