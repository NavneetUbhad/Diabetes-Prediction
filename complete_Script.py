# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 22:02:47 2023

@author: navne
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv(r"C:\Users\navne\OneDrive\Desktop\New folder\Health\diabetes.csv")
df.head()

df.shape
df.describe()

df['Outcome'].value_counts()

x=df.drop(columns='Outcome',axis=1)
y=df['Outcome']

scaler = StandardScaler()
scaler.fit(x)
StandardScaler()
standardized_data = scaler.transform(x)
standardized_data
x = standardized_data
y
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
x_train.shape
x_test.shape
clf = svm.SVC(kernel='linear')
clf.fit(x_train,y_train)

x_train_prediction = clf.predict(x_train)
accuracy_score(x_train_prediction,y_train)

x_test_prediction = clf.predict(x_test)
accuracy_score(x_test_prediction,y_test)
joblib.dump(clf, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

Pregnancies = int(input("enter number of time pregnant :-> "))
Glucose = int(input("enter your Glucose level :->"))
BloodPressure = int(input("enter your BloodPressure level :->"))
SkinThickness = int(input("enter values of SkinThickness :->"))
Insulin = int(input("enter values of Insulin :->"))
BMI = float(input("enter the value of BMI :->"))
DiabetesPedigreeFunction = float(input("enter the values of Diabetes Pedigree Function :->"))
Age = int(input("Enter your age :->"))


input_sample = (Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
input_np_array = np.asarray(input_sample)

input_np_array_reshaped = input_np_array.reshape(1,-1)
std_data = scaler.transform(input_np_array_reshaped)
std_data

prediction = clf.predict(std_data)
prediction


if (prediction[0]==0):
    print("Person is not diabetic")
else:
    print("Person is diabetic")