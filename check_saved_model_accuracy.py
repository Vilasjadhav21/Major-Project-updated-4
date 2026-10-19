from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import EHGClassifier

MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "ehg_classifier.joblib"
TEST_PATH = PROJECT_ROOT / "outputs" / "features" / "test_dataset.csv"

FEATURES = [
    "sampen_imf1",
    "median_freq_imf1_hz",
    "rms_imf1",
    "rms_imf2",
    "hilbert_par_imf1",
]

# Check that required files exist.
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not TEST_PATH.exists():
    raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")

test_df = pd.read_csv(TEST_PATH)

required_columns = FEATURES + ["label"]
missing = [col for col in required_columns if col not in test_df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

if test_df[required_columns].isna().any().any():
    raise ValueError("Missing values found. Inspect the test dataset first.")

X_test = test_df[FEATURES]
y_test = test_df["label"].astype(int)

# Load the existing trained classifier.
model = EHGClassifier.load(MODEL_PATH)

# Use the project's existing prediction method.
y_pred = model.predict(X_test)

print("\n========== OBJECTIVE 4 MODEL EVALUATION ==========")
print(f"Test records: {len(test_df)}")
print(f"Term records: {(y_test == 0).sum()}")
print(f"Preterm records: {(y_test == 1).sum()}")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(
    f"Balanced accuracy: "
    f"{balanced_accuracy_score(y_test, y_pred) * 100:.2f}%"
)
print(
    f"Preterm precision: "
    f"{precision_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}"
)
print(
    f"Preterm sensitivity: "
    f"{recall_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}"
)
print(
    f"Preterm F1-score: "
    f"{f1_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}"
)

tn, fp, fn, tp = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
).ravel()

specificity = tn / (tn + fp) if (tn + fp) else float("nan")

print(f"Specificity: {specificity:.4f}")
print("\nConfusion matrix (actual rows, predicted columns):")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Term", "Preterm"],
        zero_division=0,
    )
)

# Calculate ROC-AUC only when both classes are present.
if y_test.nunique() == 2:
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
        print(f"ROC-AUC: {roc_auc_score(y_test, scores):.4f}")
    else:
        print("ROC-AUC unavailable: model has no predict_proba method.")
else:
    print("ROC-AUC unavailable: test set contains only one class.")