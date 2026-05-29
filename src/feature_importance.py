import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/crypto_volatility_model.pkl")

# Load features
X = pd.read_csv("data/processed/X.csv")

# Feature importance
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)

plt.figure(figsize=(10,6))
plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()