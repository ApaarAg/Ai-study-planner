import pandas as pd
import os
from app.logger import LOG_PATH


def compute_metrics():

    # File does not exist
    if not os.path.isfile(LOG_PATH):
        return {
            "total_records": 0,
            "avg_normalized_gain": 0,
            "avg_predicted_gain": 0,
            "avg_delta_minutes": 0
        }

    # File exists but empty
    if os.path.getsize(LOG_PATH) == 0:
        return {
            "total_records": 0,
            "avg_normalized_gain": 0,
            "avg_predicted_gain": 0,
            "avg_delta_minutes": 0
        }

    try:
        df = pd.read_csv(LOG_PATH)
    except Exception:
        return {
            "total_records": 0,
            "avg_normalized_gain": 0,
            "avg_predicted_gain": 0,
            "avg_delta_minutes": 0
        }

    if df.empty:
        return {
            "total_records": 0,
            "avg_normalized_gain": 0,
            "avg_predicted_gain": 0,
            "avg_delta_minutes": 0
        }

    avg_norm = df["normalized_gain"].mean() if "normalized_gain" in df.columns else 0
    avg_pred = df["predicted_gain"].mean() if "predicted_gain" in df.columns else 0
    avg_delta = df["delta_minutes"].mean() if "delta_minutes" in df.columns else 0

    return {
        "total_records": len(df),
        "avg_normalized_gain": float(avg_norm),
        "avg_predicted_gain": float(avg_pred),
        "avg_delta_minutes": float(avg_delta)
    }