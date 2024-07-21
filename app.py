import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the saved label encoders
label_encoders = {}
for column in ['Airline_Company', 'Departure_City', 'Arrival_City', 'Departure_Time_Category']:
    with open(f'label_encoder_{column}.pkl', 'rb') as file:
        label_encoders[column] = pickle.load(file)

# Load the manually encoded mappings
with open('flight_class_mapping.pkl', 'rb') as file:
    flight_class_mapping = pickle.load(file)

with open('total_stops_mapping.pkl', 'rb') as file:
    total_stops_mapping = pickle.load(file)

# Load the trained Linear Regression model
with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Helper functions
def is_weekend(date_str):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    day_of_week = date_obj.weekday()
    return 1 if day_of_week >= 4 else 0

# Streamlit UI
st.title('Flight Fare Prediction')

# Date of Journey input
date_of_journey = st.date_input('Select Date of Journey')
date_of_journey_str = date_of_journey.strftime("%Y-%m-%d")
month_of_journey = date_of_journey.month
journey_day = date_of_journey.day
weekend = is_weekend(date_of_journey_str)
difference_in_days = (date_of_journey - datetime.strptime('2023-01-15', "%Y-%m-%d").date()).days

# Departure City input
departure_city = st.selectbox('Select Departure City', label_encoders['Departure_City'].classes_)
departure_city_encoded = label_encoders['Departure_City'].transform([departure_city])[0]

# Departure Time Category input
departure_time_category = st.selectbox('Select Departure Time Category', label_encoders['Departure_Time_Category'].classes_)
departure_time_category_encoded = label_encoders['Departure_Time_Category'].transform([departure_time_category])[0]

# Arrival City input (filter out the selected Departure City)
arrival_city_options = [city for city in label_encoders['Arrival_City'].classes_ if city != departure_city]
arrival_city = st.selectbox('Select Arrival City', arrival_city_options)
arrival_city_encoded = label_encoders['Arrival_City'].transform([arrival_city])[0]


# Airline Company input
airline_company = st.selectbox('Select Airline Company', label_encoders['Airline_Company'].classes_)
airline_company_encoded = label_encoders['Airline_Company'].transform([airline_company])[0]

# Flight Class input
flight_class = st.selectbox('Select Flight Class', list(flight_class_mapping.keys()))
flight_class_encoded = flight_class_mapping[flight_class]

# Total Stops input
total_stops = st.selectbox('Select Total Stops', list(total_stops_mapping.keys()))
total_stops_encoded = total_stops_mapping[total_stops]

# Prepare the input data for prediction
input_data = np.array([[month_of_journey, airline_company_encoded, flight_class_encoded, departure_city_encoded, arrival_city_encoded, 
                        total_stops_encoded, weekend, difference_in_days, departure_time_category_encoded, journey_day]])

# Prediction
if st.button('Predict Fare'):
    price_prediction = model.predict(input_data)[0]
    st.write(f'Predicted Flight Fare: {price_prediction:.2f}')