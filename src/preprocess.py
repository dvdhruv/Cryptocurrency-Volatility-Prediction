import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

df = pd.read_csv("data/dataset.csv")

# Basic cleaning

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["crypto_name", "date"]).reset_index(drop=True)

# Remove duplicates if any
df = df.drop_duplicates()

# Feature engineering

# Daily return
df["daily_return"] = df.groupby("crypto_name")["close"].pct_change()

# Price range
df["price_range"] = df["high"] - df["low"]

# Open-close change
df["open_close_change"] = df["close"] - df["open"]

# Liquidity ratio
df["volume_marketcap_ratio"] = df["volume"] / df["marketCap"].replace(0, np.nan)

# Rolling features per crypto
df["rolling_mean_7"] = df.groupby("crypto_name")["close"].transform(lambda x: x.rolling(window=7).mean())
df["rolling_std_7"] = df.groupby("crypto_name")["close"].transform(lambda x: x.rolling(window=7).std())
df["rolling_mean_volume_7"] = df.groupby("crypto_name")["volume"].transform(lambda x: x.rolling(window=7).mean())

# Realized volatility using returns
df["realized_volatility_7"] = df.groupby("crypto_name")["daily_return"].transform(
    lambda x: x.rolling(window=7).std()
)

# Next-day volatility as target
df["target_volatility"] = df.groupby("crypto_name")["realized_volatility_7"].shift(-1)

# Drop rows created by rolling and shift operations
df = df.dropna().reset_index(drop=True)

# Select features

feature_cols = [
    "open", "high", "low", "close", "volume", "marketCap",
    "daily_return", "price_range", "open_close_change",
    "volume_marketcap_ratio", "rolling_mean_7", "rolling_std_7",
    "rolling_mean_volume_7", "realized_volatility_7"
]

X = df[feature_cols]
y = df["target_volatility"]


# Scale numerical features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save processed data
os.makedirs("data/processed", exist_ok=True)
pd.DataFrame(X_scaled, columns=feature_cols).to_csv("data/processed/X.csv", index=False)
y.to_csv("data/processed/y.csv", index=False)

# Save scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

# Save cleaned dataframe too
df.to_csv("data/processed/cleaned_data.csv", index=False)

print("Preprocessing completed successfully.")
print("Shape of X:", X_scaled.shape)
print("Shape of y:", y.shape)