import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import yaml

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def evaluate(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    }

def train():
    params = load_params()
    FEATURES = params["features"]
    TARGET   = params["target"]
    
    df = pd.read_csv("data/processed/cumin_processed.csv", parse_dates=["date"])
    train = df[df["year"] < 2025].dropna(subset=FEATURES)
    test  = df[df["year"] == 2025].dropna(subset=FEATURES)

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test,  y_test  = test[FEATURES],  test[TARGET]

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("cumin-price-forecasting")

    with mlflow.start_run(run_name="XGBoost_production"):
        model_params = params["xgboost"]
        model = XGBRegressor(**model_params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        preds   = model.predict(X_test)
        metrics = evaluate(y_test, preds)

        mlflow.log_params(model_params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model",
                                registered_model_name="cumin-forecaster")

        print(f"RMSE: {metrics['rmse']:.0f} | MAE: {metrics['mae']:.0f} | MAPE: {metrics['mape']:.2f}%")
        print("Model registered as cumin-forecaster ✅")

if __name__ == "__main__":
    train()