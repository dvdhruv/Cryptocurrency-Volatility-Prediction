# Final Report

## Problem Statement
Predict cryptocurrency volatility using historical market data.

## Approach
- cleaned dataset
- engineered time-series features
- trained XGBoost regression model
- evaluated using MAE, RMSE, and R²

## Results
- MAE: 0.006736
- RMSE: 0.015673
- R²: 0.887387

## Key Insights
- realized_volatility_7 was the most important feature
- open and rolling_mean_7 were also highly useful
- liquidity features added predictive value

## Conclusion
The model explains a large portion of volatility variation and is suitable as a student-level financial forecasting project.
