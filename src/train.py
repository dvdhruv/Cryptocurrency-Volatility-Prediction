import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Optional: try XGBoost first, fallback to RandomForest
try:
    from xgboost import XGBRegressor
    use_xgb = True
except ImportError:
    use_xgb = False


# Load processed data

X = pd.read_csv("data/processed/X.csv")
y = pd.read_csv("data/processed/y.csv").values.ravel()

print("X shape:", X.shape)
print("y shape:", y.shape)

# Time-safe split

split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Model training

if use_xgb:
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    print("Using XGBoost model")
else:
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    print("Using RandomForest model")

model.fit(X_train, y_train)


# Predictions

y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R2   : {r2:.6f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/crypto_volatility_model.pkl")

print("\nModel saved successfully at models/crypto_volatility_model.pkl")