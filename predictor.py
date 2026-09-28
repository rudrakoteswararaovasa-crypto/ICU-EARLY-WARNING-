from pathlib import Path
import numpy as np
import joblib
from src.data_utils import FEATURES

def heuristic_probability(x):
    # Demo-only fallback. Not clinically validated.
    hr = x["heart_rate"]
    sbp = x["systolic_bp"]
    rr = x["respiratory_rate"]
    spo2 = x["spo2"]
    lactate = x["lactate"]
    creat = x["creatinine"]
    urine = x["urine_output"]
    age = x["age"]

    score = (
        0.018 * (hr - 85)
        - 0.028 * (sbp - 120)
        + 0.075 * (rr - 18)
        - 0.18 * (spo2 - 95)
        + 0.65 * (lactate - 1.5)
        + 0.45 * (creat - 1)
        - 0.018 * (urine - 60)
        + 0.012 * (age - 50)
    )
    return float(1 / (1 + np.exp(-(score - 1.0))))

def predict_risk(features, model_path):
    if Path(model_path).exists():
        model = joblib.load(model_path)
        row = [[features[f] for f in FEATURES]]
        p = float(model.predict_proba(row)[0, 1])
        return p, "trained scikit-learn baseline"
    return heuristic_probability(features), "demo fallback heuristic"

def risk_category(p):
    if p >= 0.70:
        return "High"
    if p >= 0.35:
        return "Medium"
    return "Low"
