from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)

from src.data_utils import generate_synthetic_dataset, FEATURES
from src.model import build_baseline

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "synthetic_icu_demo.csv"
MODEL = ROOT / "models" / "risk_model.joblib"

def main():
    df = generate_synthetic_dataset()
    DATA.parent.mkdir(exist_ok=True)
    df.to_csv(DATA, index=False)

    X = df[FEATURES]
    y = df["deterioration"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_baseline()
    model.fit(X_train, y_train)

    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)

    print("Evaluation on synthetic demo data")
    print("---------------------------------")
    print(f"AUROC:      {roc_auc_score(y_test, p):.4f}")
    print(f"AUPRC:      {average_precision_score(y_test, p):.4f}")
    print(f"Accuracy:   {accuracy_score(y_test, pred):.4f}")
    print(f"Precision:  {precision_score(y_test, pred, zero_division=0):.4f}")
    print(f"Recall:     {recall_score(y_test, pred, zero_division=0):.4f}")
    print(f"F1:         {f1_score(y_test, pred, zero_division=0):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred))

    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL)
    print(f"\nSaved model to: {MODEL}")

if __name__ == "__main__":
    main()
