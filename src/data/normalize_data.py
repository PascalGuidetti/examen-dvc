import pandas as pd
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
print(X_train.head())
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaler, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaler, columns=X_test.columns)

X_train_scaled.to_csv("data/processed/X_train_scaled.csv", index=False)
X_test_scaled.to_csv("data/processed/X_test_scaled.csv", index=False)