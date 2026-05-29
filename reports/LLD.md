# Low-Level Design (LLD)

## preprocess.py
- loads raw CSV
- converts date column
- removes duplicates
- creates engineered features
- creates target volatility
- scales inputs
- saves processed files

## train.py
- loads processed data
- splits data chronologically
- trains XGBoost model
- evaluates metrics:
  - MAE: 0.006736
  - RMSE: 0.015673
  - R²: 0.887387
- saves model

## app.py
- loads trained model and scaler
- receives user inputs
- scales values
- predicts volatility
- shows output in Streamlit
