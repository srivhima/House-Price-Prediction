import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

# Title
st.title("House Price Prediction App")

# Inputs
area = st.number_input("Enter Area (sq ft)")
bedrooms = st.number_input("Enter Bedrooms", step=1)
bathrooms = st.number_input("Enter Bathrooms", step=1)

# Prediction
if st.button("Predict Price"):

    features = np.array([[area, bedrooms, bathrooms]])

    prediction = model.predict(features)

    st.success(f"Estimated House Price: ₹ {prediction[0]:,.2f}")