# Cryptocurrency Volatility Prediction

## Dataset
Daily records for multiple cryptocurrencies including:
- date
- symbol
- open
- high
- low
- close
- volume
- marketCap

## Objective
Forecast volatility variations to support risk management and decision-making.

## Methodology
1. Data loading and cleaning
2. Feature engineering
3. Chronological train-test split
4. XGBoost training
5. Evaluation with MAE, RMSE, and R²
6. Streamlit deployment

## Engineered Features
- daily_return
- price_range
- open_close_change
- volume_marketcap_ratio
- rolling_mean_7
- rolling_std_7
- rolling_mean_volume_7
- realized_volatility_7

## Evaluation
- MAE: 0.006736
- RMSE: 0.015673
- R² Score: 0.887387

## How to Run

pip install -r requirements.txt

python src/preprocess.py

python src/train.py

python src/feature_importance.py

streamlit run app/app.py

## Author

Author: Dhruv Varshney

Project: Cryptocurrency Volatility Prediction Using Machine Learning

Technologies: Python, Pandas, Scikit-Learn, XGBoost, Streamlit
