import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from src.predictor import predict_risk, risk_category
from src.data_utils import make_demo_patient

st.set_page_config(page_title="ICU Early-Warning AI", page_icon="🏥", layout="wide")

MODEL_PATH = Path("models/risk_model.joblib")

st.title("🏥 ICU Patient Deterioration Early-Warning Prototype")
st.caption("Academic/research prototype — not a medical diagnostic system.")

st.info(
    "Use de-identified data only. The displayed risk is illustrative unless the model has been trained "
    "on an approved clinical dataset and properly validated."
)

if not MODEL_PATH.exists():
    st.warning("No trained model found. Using the built-in demo scoring model. Run `python train.py` to train the project model.")

demo = make_demo_patient()

with st.sidebar:
    st.header("Patient / Observation")
    patient_code = st.text_input("Patient Code", "ICU-DEMO-001")
    horizon = st.selectbox("Prediction Horizon", ["Next 6 hours", "Next 12 hours", "Next 24 hours"], index=1)

    st.subheader("Vital Signs")
    heart_rate = st.number_input("Heart Rate (bpm)", 30.0, 220.0, float(demo["heart_rate"]))
    systolic_bp = st.number_input("Systolic BP (mmHg)", 50.0, 220.0, float(demo["systolic_bp"]))
    respiratory_rate = st.number_input("Respiratory Rate (/min)", 5.0, 60.0, float(demo["respiratory_rate"]))
    spo2 = st.number_input("SpO₂ (%)", 50.0, 100.0, float(demo["spo2"]))
    temperature = st.number_input("Temperature (°C)", 30.0, 43.0, float(demo["temperature"]))

    st.subheader("Laboratory / Clinical")
    lactate = st.number_input("Lactate (mmol/L)", 0.0, 20.0, float(demo["lactate"]))
    wbc = st.number_input("WBC (10^9/L)", 0.1, 100.0, float(demo["wbc"]))
    creatinine = st.number_input("Creatinine (mg/dL)", 0.1, 20.0, float(demo["creatinine"]))
    age = st.number_input("Age (years)", 18, 110, int(demo["age"]))
    urine_output = st.number_input("Urine Output (mL/hr)", 0.0, 500.0, float(demo["urine_output"]))

features = {
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
}

if st.button("🔎 Predict Deterioration Risk", type="primary", use_container_width=True):
    probability, method = predict_risk(features, MODEL_PATH)
    category = risk_category(probability)

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Probability", f"{probability*100:.1f}%")
    c2.metric("Risk Category", category)
    c3.metric("Prediction Horizon", horizon)

    if category == "High":
        st.error("⚠️ Review required — high-risk category in this prototype.")
    elif category == "Medium":
        st.warning("⚠️ Review required — medium-risk category in this prototype.")
    else:
        st.success("Low-risk category in this prototype.")

    st.subheader("Current Patient Data")
    st.dataframe(pd.DataFrame([features]), use_container_width=True)

    st.caption(f"Model method: {method}. Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    st.subheader("Important interpretation")
    st.write(
        "This output is a risk estimate for an academic prototype. It is not a diagnosis, "
        "does not prescribe treatment, and should not be used to make real clinical decisions."
    )

st.divider()
st.subheader("System workflow")
st.markdown(
    "EHR/ICU data → preprocessing → temporal representation → pretrained/transfer-learning model → "
    "risk probability → validated threshold → dashboard/alert"
)
