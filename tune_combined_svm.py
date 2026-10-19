"""
Class Imbalance Management and RBF-SVM Hyperparameter Tuning for Combined Cohort.
Generates:
  - outputs/features/combined_svm_tuning_report.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from src.config import FEATURES_DIR


def evaluate_cv(pipeline, X, y, cv):
    """Evaluate pipeline across cross-validation folds."""
    accs, bal_accs, recs, specs, precs, f1s, aucs = [], [], [], [], [], [], []
    
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_val)
        
        try:
            y_prob = pipeline.predict_proba(X_val)[:, 1]
            auc_val = roc_auc_score(y_val, y_prob)
        except Exception:
            auc_val = np.nan
            
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
        spec_val = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        accs.append(accuracy_score(y_val, y_pred))
        bal_accs.append(balanced_accuracy_score(y_val, y_pred))
        recs.append(recall_score(y_val, y_pred, pos_label=1, zero_division=0))
        specs.append(spec_val)
        precs.append(precision_score(y_val, y_pred, pos_label=1, zero_division=0))
        f1s.append(f1_score(y_val, y_pred, pos_label=1, zero_division=0))
        aucs.append(auc_val)
        
    return {
        "accuracy": (np.mean(accs), np.std(accs)),
        "balanced_accuracy": (np.mean(bal_accs), np.std(bal_accs)),
        "sensitivity": (np.mean(recs), np.std(recs)),
        "specificity": (np.mean(specs), np.std(specs)),
        "precision": (np.mean(precs), np.std(precs)),
        "f1": (np.mean(f1s), np.std(f1s)),
        "roc_auc": (np.nanmean(aucs), np.nanstd(aucs))
    }


def run_tuning():
    train_csv = FEATURES_DIR / "train_dataset.csv"
    df_train = pd.read_csv(train_csv)
    
    feature_cols = [
        "sampen_imf1",
        "median_freq_imf1_hz",
        "rms_imf1",
        "rms_imf2",
        "hilbert_par_imf1"
    ]
    
    X_train = df_train[feature_cols].values
    y_train = df_train["label"].values
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("=== PHASE 8: Evaluating Imbalance Management Strategies ===")
    strategies = {
        "Strategy A (Class-Weighted Only)": ImbPipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42))
        ]),
        "Strategy B (RandomOversampling + Weighted SVM)": ImbPipeline([
            ("ros", RandomOverSampler(random_state=42)),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42))
        ]),
        "Strategy C (SMOTE + Weighted SVM)": ImbPipeline([
            ("smote", SMOTE(k_neighbors=5, random_state=42)),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42))
        ])
    }
    
    strat_results = {}
    for name, pipe in strategies.items():
        res = evaluate_cv(pipe, X_train, y_train, cv)
        strat_results[name] = res
        print(f"  {name}: BalAcc = {res['balanced_accuracy'][0]:.4f} +/- {res['balanced_accuracy'][1]:.4f}, Sens = {res['sensitivity'][0]:.4f}, Spec = {res['specificity'][0]:.4f}")
        
    # Select best imbalance strategy based on CV Balanced Accuracy
    best_strat_name = max(strat_results.keys(), key=lambda k: strat_results[k]["balanced_accuracy"][0])
    print(f"\nBest Imbalance Strategy: {best_strat_name}")
    
    # --------------------------------------------------------------------------
    # PHASE 9: Tuning Hyperparameters (C and gamma)
    # --------------------------------------------------------------------------
    print("\n=== PHASE 9: Tuning RBF-SVM Hyperparameters ===")
    c_candidates = [0.01, 0.1, 1, 10, 100, 1000]
    gamma_candidates = ["scale", 0.0001, 0.001, 0.01, 0.1, 1]
    
    tuning_records = []
    
    for c_val in c_candidates:
        for g_val in gamma_candidates:
            # Build pipeline using best imbalance strategy
            if "SMOTE" in best_strat_name:
                pipe = ImbPipeline([
                    ("smote", SMOTE(k_neighbors=5, random_state=42)),
                    ("scaler", StandardScaler()),
                    ("svm", SVC(C=c_val, gamma=g_val, kernel="rbf", class_weight="balanced", probability=True, random_state=42))
                ])
            elif "RandomOversampling" in best_strat_name:
                pipe = ImbPipeline([
                    ("ros", RandomOverSampler(random_state=42)),
                    ("scaler", StandardScaler()),
                    ("svm", SVC(C=c_val, gamma=g_val, kernel="rbf", class_weight="balanced", probability=True, random_state=42))
                ])
            else:
                pipe = ImbPipeline([
                    ("scaler", StandardScaler()),
                    ("svm", SVC(C=c_val, gamma=g_val, kernel="rbf", class_weight="balanced", probability=True, random_state=42))
                ])
                
            res = evaluate_cv(pipe, X_train, y_train, cv)
            tuning_records.append({
                "C": c_val,
                "gamma": g_val,
                "bal_acc_mean": res["balanced_accuracy"][0],
                "bal_acc_std": res["balanced_accuracy"][1],
                "acc_mean": res["accuracy"][0],
                "sens_mean": res["sensitivity"][0],
                "spec_mean": res["specificity"][0],
                "prec_mean": res["precision"][0],
                "f1_mean": res["f1"][0],
                "auc_mean": res["roc_auc"][0]
            })
            
    tuning_df = pd.DataFrame(tuning_records)
    best_config = tuning_df.sort_values(by="bal_acc_mean", ascending=False).iloc[0]
    
    print("\nBest Hyperparameter Configuration:")
    print(f"  C = {best_config['C']}, gamma = {best_config['gamma']}")
    print(f"  CV Balanced Accuracy: {best_config['bal_acc_mean']:.4f} +/- {best_config['bal_acc_std']:.4f}")
    print(f"  CV Sensitivity:       {best_config['sens_mean']:.4f}")
    print(f"  CV Specificity:       {best_config['spec_mean']:.4f}")
    
    # Save combined tuning report
    out_txt = FEATURES_DIR / "combined_svm_tuning_report.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("        OBJECTIVE 4: IMBALANCE MANAGEMENT & RBF-SVM TUNING REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. TRAINING COHORT CLASS DISTRIBUTION (N=260):\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Term Records (0):     {(y_train == 0).sum()} (84.62%)\n")
        f.write(f"  Preterm Records (1):  {(y_train == 1).sum()} (15.38%)\n")
        f.write(f"  Class Imbalance:      {((y_train == 0).sum()) / ((y_train == 1).sum()):.2f}:1\n")
        f.write("  Cross-Validation:     5-Fold StratifiedKFold (random_state=42)\n\n")
        
        f.write("2. PHASE 8: IMBALANCE MANAGEMENT STRATEGY COMPARISON:\n")
        f.write("-" * 80 + "\n")
        for sname, sres in strat_results.items():
            f.write(f"  {sname}:\n")
            f.write(f"    * Balanced Accuracy: {sres['balanced_accuracy'][0]:.4f} +/- {sres['balanced_accuracy'][1]:.4f}\n")
            f.write(f"    * Preterm Recall:    {sres['sensitivity'][0]:.4f} +/- {sres['sensitivity'][1]:.4f}\n")
            f.write(f"    * Term Specificity:  {sres['specificity'][0]:.4f} +/- {sres['specificity'][1]:.4f}\n")
            f.write(f"    * Precision:         {sres['precision'][0]:.4f} +/- {sres['precision'][1]:.4f}\n")
            f.write(f"    * F1-Score:          {sres['f1'][0]:.4f} +/- {sres['f1'][1]:.4f}\n")
            f.write(f"    * ROC-AUC:           {sres['roc_auc'][0]:.4f} +/- {sres['roc_auc'][1]:.4f}\n\n")
        f.write(f"  Selected Imbalance Strategy: {best_strat_name}\n\n")
        
        f.write("3. PHASE 9: RBF-SVM HYPERPARAMETER GRID TUNING:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Grid Search Range: C in {c_candidates}, gamma in {gamma_candidates}\n\n")
        f.write("Top 10 Configurations by 5-Fold Cross-Validation Balanced Accuracy:\n")
        top10 = tuning_df.sort_values(by="bal_acc_mean", ascending=False).head(10)
        f.write(top10.to_string(index=False) + "\n\n")
        
        f.write("4. FINAL SELECTED HYPERPARAMETER CONFIGURATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Imbalance Strategy:   {best_strat_name}\n")
        f.write(f"  C:                    {best_config['C']}\n")
        f.write(f"  gamma:                {best_config['gamma']}\n")
        f.write(f"  CV Balanced Accuracy: {best_config['bal_acc_mean']:.4f} +/- {best_config['bal_acc_std']:.4f}\n")
        f.write(f"  CV Overall Accuracy:  {best_config['acc_mean']:.4f}\n")
        f.write(f"  CV Preterm Sensitivity:{best_config['sens_mean']:.4f}\n")
        f.write(f"  CV Term Specificity:  {best_config['spec_mean']:.4f}\n")
        f.write(f"  CV Precision:         {best_config['prec_mean']:.4f}\n")
        f.write(f"  CV F1-score:          {best_config['f1_mean']:.4f}\n")
        f.write(f"  CV ROC-AUC:           {best_config['auc_mean']:.4f}\n")
        f.write("=" * 80 + "\n")
        
    print(f"Tuning report written to {out_txt}")
    return best_strat_name, best_config


if __name__ == "__main__":
    run_tuning()
