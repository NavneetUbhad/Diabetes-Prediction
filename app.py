# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 22:08:57 2023

@author: navne
"""

from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
clf = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input values
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = int(request.form['Glucose'])
        BloodPressure = int(request.form['BloodPressure'])
        SkinThickness = int(request.form['SkinThickness'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = int(request.form['Age'])

        # Create an input sample as a numpy array
        input_sample = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

        # Scale the input data using the saved scaler
        input_sample_reshaped = input_sample.reshape(1, -1)
        std_data = scaler.transform(input_sample_reshaped)

        # Make a prediction using the loaded model
        prediction = clf.predict(std_data)

        # Display the prediction result
        if prediction[0] == 0:
            result = "Person is not diabetic"
        else:
            result = "Person is diabetic"

        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
