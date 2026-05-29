# High-Level Design (HLD)

## System Overview
The system predicts cryptocurrency volatility from historical market data using a machine learning pipeline.

## Architecture
Dataset -> Preprocessing -> Feature Engineering -> Scaling -> Model Training -> Evaluation -> Streamlit Prediction App

## Components
### Data Layer
Historical OHLC, volume, and market cap data.

### Processing Layer
- clean data
- sort by crypto and date
- generate rolling and liquidity features

### Modeling Layer
XGBoost regression model predicts next-day volatility.

### UI Layer
Streamlit app accepts market inputs and returns a volatility prediction.
