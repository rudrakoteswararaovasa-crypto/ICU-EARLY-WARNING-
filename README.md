# ICU Early-Warning Risk Prediction Prototype

This project implements a runnable academic prototype based on the supplied project documentation:

**Early Disease Risk Prediction from Electronic Health Records via Transfer Learning from Pretrained Medical Foundation Models, Applied to ICU Patient Deterioration Early-Warning Systems**

## What is included

- Streamlit web dashboard
- Synthetic ICU-style demo dataset
- Data generation and preprocessing pipeline
- Baseline logistic-regression model
- Evaluation metrics: AUROC, AUPRC, precision, recall, F1, confusion matrix
- FastAPI prediction API
- Model saving/loading with joblib
- Clear extension points for a real pretrained medical representation model

## Important

The supplied documentation specifies transfer learning from a pretrained medical/clinical model, but it does not select one exact pretrained model or provide a clinical dataset. Therefore this runnable version uses a transparent synthetic-data baseline so the website works immediately. It should NOT be described as a clinically validated medical foundation model.

For an academic implementation, replace the synthetic data with an approved de-identified ICU dataset and add the selected pretrained model after fixing the prediction outcome and horizon.

## Windows setup

```powershell
cd icu_early_warning
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

The browser normally opens at:
http://localhost:8501

## API

In another terminal:

```powershell
uvicorn api:app --reload
```

API docs:
http://127.0.0.1:8000/docs

## Project structure

```text
icu_early_warning/
├── app.py
├── api.py
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
├── data/
│   └── synthetic_icu_demo.csv
├── models/
│   └── risk_model.joblib
├── reports/
└── src/
    ├── __init__.py
    ├── data_utils.py
    ├── model.py
    └── predictor.py
```

## Suggested next academic step

Implement the source-documentation workflow:

EHR data → preprocessing → temporal features → pretrained medical/clinical representation → fine-tuning → risk probability → validated threshold → dashboard.

Patient-level train/validation/test splitting and leakage prevention are required for a real evaluation.


## Simple website option (works with FastAPI)

If Streamlit is unavailable on your machine, use the included HTML website:

```powershell
uvicorn web:app --reload
```

Open:
http://127.0.0.1:8000

This page uses the same `/predict` backend and trained model.
