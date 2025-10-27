import pickle
import pandas as pd
import numpy as np
import json
import os
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

print("Evaluation du modèle entraîné...")
X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv")
y_test = np.ravel(y_test)

with open("models/xgb_trained_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

y_pred = xgb_model.predict(X_test)

df = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred
})

file_path = "data/predictions/"
os.makedirs(file_path, exist_ok=True)
df.to_csv(file_path + "prediction.csv", index=False)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

metrics = {"R2": r2, "MSE": mse}

file_path = "metrics/"
os.makedirs(file_path, exist_ok=True)
with open(file_path + "score.json", "w") as f:
    json.dump(metrics, f)
