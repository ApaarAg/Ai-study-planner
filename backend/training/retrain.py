import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import json
import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "session_logs.csv")
MODEL_PATH = os.path.join(BASE_DIR, "app", "model", "mastery_model.pkl")
EXPERIMENT_PATH = os.path.join(BASE_DIR, "experiments")

os.makedirs(EXPERIMENT_PATH, exist_ok=True)


def retrain_model():

    if not os.path.isfile(DATA_PATH):
        print("No session data found.")
        return

    df = pd.read_csv(DATA_PATH)

    if len(df) < 200:
        print("Not enough data to retrain.")
        return

    FEATURES = [
        "difficulty",
        "past_score",
        "remaining_gap",
        "hours_spent",
        "revision_count",
        "days_to_exam",
        "confidence",
        "predicted_minutes"
    ]

    if "normalized_gain" not in df.columns:
        print("Missing normalized_gain column.")
        return

    df["remaining_gap"] = 100 - df["past_score"]

    X = df[FEATURES]
    y = df["normalized_gain"]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05
    )

    model.fit(X, y)

    preds = model.predict(X)

    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))

    joblib.dump(model, MODEL_PATH)

    experiment = {
        "timestamp": str(datetime.datetime.now()),
        "samples": int(len(df)),
        "r2": float(r2),
        "rmse": float(rmse)
    }

    with open(os.path.join(EXPERIMENT_PATH, "training_log.json"), "a") as f:
        f.write(json.dumps(experiment) + "\n")

    print("Retraining complete.")
    print("R2:", r2)
    print("RMSE:", rmse)


if __name__ == "__main__":
    retrain_model()