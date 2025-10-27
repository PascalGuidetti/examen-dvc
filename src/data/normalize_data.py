import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

print("Normalisation des données...")
scaler = MinMaxScaler()

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaler, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaler, columns=X_test.columns)

file_path = "data/processed/"
os.makedirs(file_path, exist_ok=True)
X_train_scaled.to_csv(file_path + "X_train_scaled.csv", index=False)
X_test_scaled.to_csv(file_path + "X_test_scaled.csv", index=False)