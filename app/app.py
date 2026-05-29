import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("models/crypto_volatility_model.pkl")
scaler = joblib.load("models/scaler.pkl")

feature_cols = [
    "open", "high", "low", "close", "volume", "marketCap",
    "daily_return", "price_range", "open_close_change",
    "volume_marketcap_ratio", "rolling_mean_7", "rolling_std_7",
    "rolling_mean_volume_7", "realized_volatility_7"
]

st.title("Cryptocurrency Volatility Prediction")
st.write("Enter feature values to predict next-day volatility.")

# Input fields
open_price = st.number_input("Open Price", value=100.0)
high_price = st.number_input("High Price", value=105.0)
low_price = st.number_input("Low Price", value=95.0)
close_price = st.number_input("Close Price", value=102.0)
volume = st.number_input("Volume", value=1000000.0)
market_cap = st.number_input("Market Cap", value=50000000.0)

daily_return = st.number_input("Daily Return", value=0.02)
price_range = high_price - low_price
open_close_change = close_price - open_price
volume_marketcap_ratio = volume / market_cap if market_cap != 0 else 0
rolling_mean_7 = st.number_input("7-Day Rolling Mean", value=101.0)
rolling_std_7 = st.number_input("7-Day Rolling Std", value=2.5)
rolling_mean_volume_7 = st.number_input("7-Day Rolling Volume Mean", value=950000.0)
realized_volatility_7 = st.number_input("7-Day Realized Volatility", value=0.03)

if st.button("Predict Volatility"):
    input_data = pd.DataFrame([[
        open_price, high_price, low_price, close_price, volume, market_cap,
        daily_return, price_range, open_close_change,
        volume_marketcap_ratio, rolling_mean_7, rolling_std_7,
        rolling_mean_volume_7, realized_volatility_7
    ]], columns=feature_cols)

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    st.success(f"Predicted Next-Day Volatility: {prediction:.6f}")