from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from src.predictor import predict_risk, risk_category

app = FastAPI(title="ICU Early-Warning Prototype API")
MODEL_PATH = Path("models/risk_model.joblib")
HTML_PATH = Path("web.html")

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

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PATH.read_text(encoding="utf-8")

@app.post("/predict")
def predict(patient: PatientObservation):
    probability, method = predict_risk(patient.model_dump(), MODEL_PATH)
    return {
        "risk_probability": round(probability, 4),
        "risk_category": risk_category(probability),
        "method": method,
        "note": "Research prototype only; not a diagnosis."
    }
