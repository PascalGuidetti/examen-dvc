import pandas as pd
import os
from sklearn.model_selection import train_test_split


print("Split des données en train et test...")
df = pd.read_csv("data/raw/raw.csv")

X = df.drop(columns=["silica_concentrate", "date"])
y = df["silica_concentrate"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

file_path = "data/processed/"
os.makedirs(file_path, exist_ok=True)
X_train.to_csv(file_path + "X_train.csv", index=False)
X_test.to_csv(file_path + "X_test.csv", index=False)
y_train.to_csv(file_path + "y_train.csv", index=False)
y_test.to_csv(file_path + "y_test.csv", index=False)