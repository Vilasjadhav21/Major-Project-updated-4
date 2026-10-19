"""
Train Final Pipeline and Evaluate on Independent Test Set (Combined Cohort).
Generates:
  - outputs/models/ehg_classifier.joblib
  - outputs/models/standard_scaler.joblib
  - outputs/features/combined_model_accuracy_report.txt
  - outputs/combined_confusion_matrix.png
  - outputs/combined_roc_curve.png
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from src.config import FEATURES_DIR, MODELS_DIR, OUTPUT_DIR


def train_and_evaluate():
    train_csv = FEATURES_DIR / "train_dataset.csv"
    test_csv = FEATURES_DIR / "test_dataset.csv"
    
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    
    feature_cols = [
        "sampen_imf1",
        "median_freq_imf1_hz",
        "rms_imf1",
        "rms_imf2",
        "hilbert_par_imf1"
    ]
    
    X_train = df_train[feature_cols].values
    y_train = df_train["label"].values
    
    X_test = df_test[feature_cols].values
    y_test = df_test["label"].values
    
    print(f"Training cohort: N={len(df_train)} (Term: {(y_train == 0).sum()}, Preterm: {(y_train == 1).sum()})")
    print(f"Testing cohort:  N={len(df_test)} (Term: {(y_test == 0).sum()}, Preterm: {(y_test == 1).sum()})")
    
    # --------------------------------------------------------------------------
    # PHASE 10: Train Final Model Pipeline
    # --------------------------------------------------------------------------
    # Apply selected Strategy B: RandomOverSampler on training data, then standard Pipeline
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    print(f"Resampled training set: N={len(X_train_resampled)} (Term: {(y_train_resampled == 0).sum()}, Preterm: {(y_train_resampled == 1).sum()})")
    
    scaler = StandardScaler()
    scaler.fit(X_train_resampled)
    
    svm_clf = SVC(
        C=1.0,
        gamma="scale",
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=42
    )
    
    # Full inference pipeline (StandardScaler + SVC)
    final_pipeline = Pipeline([
        ("scaler", scaler),
        ("svm", svm_clf)
    ])
    
    final_pipeline.fit(X_train_resampled, y_train_resampled)
    
    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "ehg_classifier.joblib"
    scaler_path = MODELS_DIR / "standard_scaler.joblib"
    
    joblib.dump(final_pipeline, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved pipeline to {model_path} and scaler to {scaler_path}")
    
    # Verify reloading
    reloaded_model = joblib.load(model_path)
    sample_preds = reloaded_model.predict(X_test[:5])
    print(f"Reload verification sample predictions: {sample_preds}")
    
    # --------------------------------------------------------------------------
    # PHASE 11: Independent Test Set Evaluation
    # --------------------------------------------------------------------------
    y_pred = reloaded_model.predict(X_test)
    y_prob = reloaded_model.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, y_pred)
    test_bal_acc = balanced_accuracy_score(y_test, y_pred)
    test_sens = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    test_prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    test_auc = roc_auc_score(y_test, y_prob)
    
    print("\n=== FINAL INDEPENDENT TEST METRICS (N=66) ===")
    print(f"  Overall Accuracy:     {test_acc * 100:.2f}%")
    print(f"  Balanced Accuracy:    {test_bal_acc * 100:.2f}%")
    print(f"  Preterm Sensitivity:  {test_sens * 100:.2f}% ({tp}/{tp + fn})")
    print(f"  Term Specificity:     {test_spec * 100:.2f}% ({tn}/{tn + fp})")
    print(f"  Preterm Precision:    {test_prec * 100:.2f}%")
    print(f"  F1-Score:             {test_f1:.4f}")
    print(f"  ROC-AUC:              {test_auc:.4f}")
    print(f"  Confusion Matrix:     TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    # Dataset subgroup breakdown
    v101_mask = (df_test["dataset_version"] == "v1.0.1").values
    v100_mask = (df_test["dataset_version"] == "v1.0.0").values
    
    def calc_subgroup(mask):
        if not any(mask):
            return {}
        y_sub_true = y_test[mask]
        y_sub_pred = y_pred[mask]
        y_sub_prob = y_prob[mask]
        cm_sub = confusion_matrix(y_sub_true, y_sub_pred, labels=[0, 1])
        s_tn, s_fp, s_fn, s_tp = cm_sub.ravel()
        return {
            "n": int(np.sum(mask)),
            "term_n": int(np.sum(y_sub_true == 0)),
            "preterm_n": int(np.sum(y_sub_true == 1)),
            "accuracy": accuracy_score(y_sub_true, y_sub_pred),
            "balanced_accuracy": balanced_accuracy_score(y_sub_true, y_sub_pred),
            "sensitivity": recall_score(y_sub_true, y_sub_pred, pos_label=1, zero_division=0),
            "specificity": s_tn / (s_tn + s_fp) if (s_tn + s_fp) > 0 else 0.0,
            "tp": s_tp, "fp": s_fp, "tn": s_tn, "fn": s_fn
        }
        
    v101_metrics = calc_subgroup(v101_mask)
    v100_metrics = calc_subgroup(v100_mask)
    
    # Save report
    report_file = FEATURES_DIR / "combined_model_accuracy_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("        OBJECTIVE 4: COMBINED DATASET MODEL ACCURACY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. COHORT COMPOSITION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Total Dataset Records:       326 (v1.0.1: 300, v1.0.0: 26)\n")
        f.write(f"  Training Cohort (N):         {len(df_train)} (Term: {(y_train == 0).sum()}, Preterm: {(y_train == 1).sum()})\n")
        f.write(f"  Independent Holdout Set (N): {len(df_test)} (Term: {tn + fp}, Preterm: {tp + fn})\n")
        f.write(f"  Model Artifact Path:         {model_path}\n")
        f.write(f"  Scaler Artifact Path:        {scaler_path}\n\n")
        
        f.write("2. FINAL INDEPENDENT TEST METRICS (FULL TEST COHORT, N=66):\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Overall Accuracy:            {test_acc * 100:.2f}% ({test_acc:.4f})\n")
        f.write(f"  Balanced Accuracy:           {test_bal_acc * 100:.2f}% ({test_bal_acc:.4f})\n")
        f.write(f"  Preterm Sensitivity:         {test_sens * 100:.2f}% ({test_sens:.4f}) [{tp}/{tp + fn} Preterms detected]\n")
        f.write(f"  Term Specificity:            {test_spec * 100:.2f}% ({test_spec:.4f}) [{tn}/{tn + fp} Terms detected]\n")
        f.write(f"  Preterm Precision:           {test_prec * 100:.2f}% ({test_prec:.4f})\n")
        f.write(f"  F1-Score:                    {test_f1:.4f}\n")
        f.write(f"  ROC-AUC:                     {test_auc:.4f}\n\n")
        
        f.write("3. CONFUSION MATRIX:\n")
        f.write("-" * 80 + "\n")
        f.write("                   Predicted Term (0)    Predicted Preterm (1)\n")
        f.write(f"True Term (0):     {tn:<22} {fp}\n")
        f.write(f"True Preterm (1):  {fn:<22} {tp}\n\n")
        f.write(f"  True Positives (TP):  {tp}\n")
        f.write(f"  False Positives (FP): {fp}\n")
        f.write(f"  True Negatives (TN):  {tn}\n")
        f.write(f"  False Negatives (FN): {fn}\n\n")
        
        f.write("4. CLASSIFICATION REPORT:\n")
        f.write("-" * 80 + "\n")
        f.write(classification_report(y_test, y_pred, target_names=["Term (0)", "Preterm (1)"]) + "\n\n")
        
        f.write("5. PERFORMANCE BREAKDOWN BY DATASET SOURCE:\n")
        f.write("-" * 80 + "\n")
        f.write(f"A. TPEHGDB v1.0.1 Test Cohort (N={v101_metrics.get('n', 0)}):\n")
        f.write(f"   - Term: {v101_metrics.get('term_n', 0)}, Preterm: {v101_metrics.get('preterm_n', 0)}\n")
        f.write(f"   - Accuracy:          {v101_metrics.get('accuracy', 0)*100:.2f}%\n")
        f.write(f"   - Balanced Accuracy: {v101_metrics.get('balanced_accuracy', 0)*100:.2f}%\n")
        f.write(f"   - Preterm Sensitivity:{v101_metrics.get('sensitivity', 0)*100:.2f}% ({v101_metrics.get('tp', 0)}/{v101_metrics.get('tp', 0)+v101_metrics.get('fn', 0)})\n")
        f.write(f"   - Term Specificity:  {v101_metrics.get('specificity', 0)*100:.2f}% ({v101_metrics.get('tn', 0)}/{v101_metrics.get('tn', 0)+v101_metrics.get('fp', 0)})\n\n")
        
        f.write(f"B. TPEHGT v1.0.0 Test Cohort (N={v100_metrics.get('n', 0)}):\n")
        f.write(f"   - Term: {v100_metrics.get('term_n', 0)}, Preterm: {v100_metrics.get('preterm_n', 0)}\n")
        f.write(f"   - Accuracy:          {v100_metrics.get('accuracy', 0)*100:.2f}%\n")
        f.write(f"   - Balanced Accuracy: {v100_metrics.get('balanced_accuracy', 0)*100:.2f}%\n")
        f.write(f"   - Preterm Sensitivity:{v100_metrics.get('sensitivity', 0)*100:.2f}% ({v100_metrics.get('tp', 0)}/{v100_metrics.get('tp', 0)+v100_metrics.get('fn', 0)})\n")
        f.write(f"   - Term Specificity:  {v100_metrics.get('specificity', 0)*100:.2f}% ({v100_metrics.get('tn', 0)}/{v100_metrics.get('tn', 0)+v100_metrics.get('fp', 0)})\n\n")
        
        f.write("6. CLINICAL DISCLAIMER:\n")
        f.write("-" * 80 + "\n")
        f.write("This research evaluation represents an empirical machine learning benchmark on PhysioNet\n")
        f.write("EHG cohorts. It does not claim clinical diagnostic validity or medical decision endorsement.\n")
        f.write("=" * 80 + "\n")
        
    print(f"Accuracy report written to {report_file}")
    
    # --------------------------------------------------------------------------
    # Plots: Confusion Matrix & ROC Curve
    # --------------------------------------------------------------------------
    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["Term (0)", "Preterm (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_ylabel("True Outcome", fontsize=12, labelpad=8)
    ax.set_xlabel("Predicted Outcome", fontsize=12, labelpad=8)
    ax.set_title(f"Confusion Matrix (N={len(y_test)})\nAccuracy: {test_acc*100:.1f}%, Bal Acc: {test_bal_acc*100:.1f}%", fontsize=12, pad=12)
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_plot_path = OUTPUT_DIR / "combined_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_plot_path}")
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"RBF-SVM ROC curve (AUC = {test_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Chance baseline (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve\nCombined EHG Cohort Holdout Test Set", fontsize=12, pad=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_plot_path = OUTPUT_DIR / "combined_roc_curve.png"
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"Saved ROC plot to {roc_plot_path}")


if __name__ == "__main__":
    train_and_evaluate()
