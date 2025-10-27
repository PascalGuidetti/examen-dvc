import pickle
import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor

print("Entraînement du modèle avec les meilleurs hyperparamètres...")
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
y_train = np.ravel(y_train)

with open("models/xgb_best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

xgb_model = XGBRegressor(**best_params)

xgb_model.fit(X_train, y_train)

file_path = "models/"
os.makedirs(file_path, exist_ok=True)
with open(file_path + "xgb_trained_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)