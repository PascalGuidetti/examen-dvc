import pandas as pd
import pickle
import os
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score

print("Grid search des hyperparametres ...")
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

xgb_model = XGBRegressor()

param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [5, 6],
    "learning_rate": [0.01, 0.05],
    "subsample": [0.8, 0.9]
}

scorer = make_scorer(r2_score)

grid = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring=scorer,
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Meilleurs hyperparamètres XGB:", grid.best_params_)

file_path = "models/"
os.makedirs(file_path, exist_ok=True)
with open(file_path + "xgb_best_params.pkl", "wb") as f:
    pickle.dump(grid.best_params_, f)
