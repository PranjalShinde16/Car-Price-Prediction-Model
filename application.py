import pickle

from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
app = Flask(__name__)

model = pickle.load(open("CarPredictionModel.pkl", 'rb'))
df = pd.read_csv("cleaned_data.csv")


@app.route('/')
def index():
    manufacturer = sorted(df['Brand'].unique())
    manufacturer.insert(0, "Select Manufacturer")
    models = sorted(df['Name'].unique())
    location = sorted(df['Location'].unique())
    year = sorted(df['Year'].unique(), reverse=True)
    fuel_type = sorted(df['Fuel_Type'].unique())
    transmission = sorted(df['Transmission'].unique())
    owner = sorted(df['Owner_Type'].unique())
    seats = sorted(df['Seats'].unique(), reverse=True)
    return render_template('predict.html', manufacturer=manufacturer, model=models, location=location, year=year, fuel_type=fuel_type, transmission=transmission, owner=owner, seats=seats)


@app.route('/predict', methods=['POST'])
def predict():
    manufacturer = request.form.get('manufacturer')
    modelsss = request.form.get('model')
    location = request.form.get('location')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    transmission = request.form.get('transmission')
    owner = request.form.get('owner')
    mileage = float(request.form.get('mileage'))
    engine = int(request.form.get('engine'))
    power = float(request.form.get('power'))
    kms_driven = int(request.form.get('kms_driven'))
    seats = int(request.form.get('seats'))

    prediction = model.predict(pd.DataFrame([[modelsss, location, year, kms_driven, fuel_type, transmission, owner, mileage, engine, power, seats, manufacturer]], columns=['Name', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats', 'Brand']))
    return str(np.round(prediction[0]), '2')


if __name__ == '__main__':
    app.run(debug=True)