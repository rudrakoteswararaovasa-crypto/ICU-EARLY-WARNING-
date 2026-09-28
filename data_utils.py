import numpy as np
import pandas as pd

FEATURES = [
    "heart_rate", "systolic_bp", "respiratory_rate", "spo2",
    "temperature", "lactate", "wbc", "creatinine", "age", "urine_output"
]

def make_demo_patient():
    return {
        "heart_rate": 92.0,
        "systolic_bp": 118.0,
        "respiratory_rate": 18.0,
        "spo2": 97.0,
        "temperature": 37.0,
        "lactate": 1.4,
        "wbc": 8.5,
        "creatinine": 1.0,
        "age": 52,
        "urine_output": 65.0,
    }

def generate_synthetic_dataset(n=2500, seed=42):
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 91, n)
    heart_rate = np.clip(rng.normal(88, 20, n), 45, 180)
    systolic_bp = np.clip(rng.normal(120, 22, n), 60, 210)
    respiratory_rate = np.clip(rng.normal(19, 6, n), 8, 45)
    spo2 = np.clip(rng.normal(96.5, 2.5, n), 75, 100)
    temperature = np.clip(rng.normal(37.0, 0.8, n), 34, 41)
    lactate = np.clip(rng.lognormal(mean=np.log(1.5), sigma=0.45, size=n), 0.5, 15)
    wbc = np.clip(rng.normal(9.5, 4.0, n), 1, 40)
    creatinine = np.clip(rng.lognormal(mean=np.log(1.0), sigma=0.45, size=n), 0.2, 10)
    urine_output = np.clip(rng.normal(65, 30, n), 0, 220)

    # Synthetic educational target: deliberately generated from clinical-like patterns.
    logit = (
        -5.0
        + 0.018 * (heart_rate - 85)
        - 0.028 * (systolic_bp - 120)
        + 0.075 * (respiratory_rate - 18)
        - 0.18 * (spo2 - 95)
        + 0.65 * (lactate - 1.5)
        + 0.015 * (wbc - 9)
        + 0.45 * (creatinine - 1)
        + 0.012 * (age - 50)
        - 0.018 * (urine_output - 60)
    )
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)

    df = pd.DataFrame({
        "heart_rate": heart_rate,
        "systolic_bp": systolic_bp,
        "respiratory_rate": respiratory_rate,
        "spo2": spo2,
        "temperature": temperature,
        "lactate": lactate,
        "wbc": wbc,
        "creatinine": creatinine,
        "age": age,
        "urine_output": urine_output,
        "deterioration": y
    })
    return df
