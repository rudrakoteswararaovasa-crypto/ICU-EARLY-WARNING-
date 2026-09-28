from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix

from src.data_utils import FEATURES, generate_synthetic_dataset

ROOT = Path(__file__).parent

def main():
    model_path = ROOT / "models" / "risk_model.joblib"
    data_path = ROOT / "data" / "synthetic_icu_demo.csv"

    if not model_path.exists():
        print("Model not found. Run: python train.py")
        return

    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        df = generate_synthetic_dataset()

    model = joblib.load(model_path)
    X, y = df[FEATURES], df["deterioration"]
    p = model.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)

    print(f"AUROC: {roc_auc_score(y, p):.4f}")
    print(f"AUPRC: {average_precision_score(y, p):.4f}")
    print("\nClassification report:")
    print(classification_report(y, pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y, pred))

if __name__ == "__main__":
    main()
