from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.predictor import predict_risk, risk_category

app = FastAPI(
    title="ICU Early-Warning Prototype API",
    version="1.0.0",
    description="Academic decision-support prototype. Not a medical diagnostic system."
)

MODEL_PATH = Path("models/risk_model.joblib")

class PatientObservation(BaseModel):
    heart_rate: float = Field(..., ge=30, le=220)
    systolic_bp: float = Field(..., ge=50, le=220)
    respiratory_rate: float = Field(..., ge=5, le=60)
    spo2: float = Field(..., ge=50, le=100)
    temperature: float = Field(..., ge=30, le=43)
    lactate: float = Field(..., ge=0, le=20)
    wbc: float = Field(..., ge=0.1, le=100)
    creatinine: float = Field(..., ge=0.1, le=20)
    age: int = Field(..., ge=18, le=110)
    urine_output: float = Field(..., ge=0, le=500)

@app.get("/")
def root():
    return {"message": "ICU Early-Warning Prototype API is running"}

@app.post("/predict")
def predict(patient: PatientObservation):
    probability, method = predict_risk(patient.model_dump(), MODEL_PATH)
    return {
        "risk_probability": round(probability, 4),
        "risk_category": risk_category(probability),
        "method": method,
        "note": "Research prototype only; not a diagnosis."
    }
