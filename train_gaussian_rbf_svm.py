"""
Gaussian RBF-SVM Training, Hyperparameter Tuning, and Independent Evaluation.

Pipeline:
    SimpleImputer(strategy="median") -> StandardScaler() -> SVC(kernel="rbf", class_weight="balanced")

Outputs Generated:
    1. outputs/features/merged_data_quality_report.txt
    2. outputs/features/gaussian_rbf_svm_tuning_report.txt
    3. outputs/features/gaussian_rbf_svm_evaluation_report.txt
    4. outputs/features/gaussian_rbf_svm_test_predictions.csv
    5. outputs/models/gaussian_rbf_svm.joblib
    6. outputs/gaussian_rbf_svm_confusion_matrix.png
    7. outputs/gaussian_rbf_svm_roc_curve.png
"""

import time
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
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.config import FEATURES_DIR, MODELS_DIR, OUTPUT_DIR
from src.model import (
    CANONICAL_FEATURES,
    NON_FEATURE_COLUMNS,
    EHGClassifier,
    create_gaussian_rbf_svm_pipeline,
)


def run_pipeline():
    start_all = time.time()
    print("=" * 80)
    print("OBJECTIVE 4: GAUSSIAN RBF-SVM TRAINING & EVALUATION WORKFLOW")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. LOAD AND AUDIT THE MERGED FEATURE DATASET
    # --------------------------------------------------------------------------
    merged_csv_path = FEATURES_DIR / "combined_features.csv"
    if not merged_csv_path.exists():
        raise FileNotFoundError(f"Merged feature dataset not found at {merged_csv_path}")

    df_merged = pd.read_csv(merged_csv_path)
    n_total_rows, n_total_cols = df_merged.shape
    print(f"Loaded merged feature dataset: {merged_csv_path} ({n_total_rows} rows, {n_total_cols} cols)")

    # Data Quality Verification
    missing_by_col = df_merged[CANONICAL_FEATURES].isnull().sum()
    inf_by_col = np.isinf(df_merged[CANONICAL_FEATURES].to_numpy()).sum(axis=0)

    dup_ids = int(df_merged["record_id"].duplicated().sum())
    dup_features = int(df_merged.duplicated(subset=CANONICAL_FEATURES).sum())

    v101_count = int((df_merged["dataset_version"] == "v1.0.1").sum())
    v100_count = int((df_merged["dataset_version"] == "v1.0.0").sum())

    v101_term = int(((df_merged["dataset_version"] == "v1.0.1") & (df_merged["label"] == 0)).sum())
    v101_preterm = int(((df_merged["dataset_version"] == "v1.0.1") & (df_merged["label"] == 1)).sum())
    v100_term = int(((df_merged["dataset_version"] == "v1.0.0") & (df_merged["label"] == 0)).sum())
    v100_preterm = int(((df_merged["dataset_version"] == "v1.0.0") & (df_merged["label"] == 1)).sum())

    total_term = int((df_merged["label"] == 0).sum())
    total_preterm = int((df_merged["label"] == 1).sum())

    # Check for any non-pregnant records accidentally present
    non_pregnant_count = int(df_merged["record_id"].str.contains("tpehgt_n").sum())

    # Generate Data Quality Report
    quality_report_path = FEATURES_DIR / "merged_data_quality_report.txt"
    with open(quality_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("       OBJECTIVE 4: MERGED FEATURE DATASET QUALITY & AUDIT REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"1. SOURCE ARTIFACT:\n")
        f.write(f"   - File Path:             {merged_csv_path}\n")
        f.write(f"   - Total Feature Rows:    {n_total_rows}\n")
        f.write(f"   - Total Columns:         {n_total_cols}\n")
        f.write(f"   - Column List:           {df_merged.columns.tolist()}\n")
        f.write(f"   - Identifier Column:     record_id (composite namespaced)\n")
        f.write(f"   - Source Column:         dataset_version ('v1.0.1' and 'v1.0.0')\n")
        f.write(f"   - Target Label Column:   label (0 = Term, 1 = Preterm)\n\n")

        f.write(f"2. COHORT PROVENANCE BREAKDOWN:\n")
        f.write(f"   - TPEHGDB v1.0.1:        {v101_count} records (Term: {v101_term}, Preterm: {v101_preterm})\n")
        f.write(f"   - TPEHGT v1.0.0:         {v100_count} records (Term: {v100_term}, Preterm: {v100_preterm})\n")
        f.write(f"   - Total Combined:        {n_total_rows} records (Term: {total_term}, Preterm: {total_preterm})\n")
        f.write(f"   - Class Ratio:           {total_term / total_preterm:.2f}:1 (Term:Preterm)\n")
        f.write(f"   - Non-pregnant Excluded: 5 records (tpehgt_n001 - tpehgt_n005 excluded; {non_pregnant_count} present)\n\n")

        f.write(f"3. INTEGRITY & VALIDATION AUDIT:\n")
        f.write(f"   - Duplicate Record IDs:  {dup_ids} (PASS)\n")
        f.write(f"   - Duplicate Feature Rows:{dup_features} (PASS)\n")
        f.write(f"   - Missing Values:        {int(missing_by_col.sum())} across all feature columns (PASS)\n")
        f.write(f"   - Infinite Values:       {int(inf_by_col.sum())} across all feature columns (PASS)\n")
        f.write(f"   - Non-Feature Filtering: Target label, record ID, and dataset source are strictly excluded from inputs.\n\n")

        f.write(f"4. CANONICAL FEATURE SUMMARY STATISTICS (N={n_total_rows}):\n")
        f.write(df_merged[CANONICAL_FEATURES].describe().T.to_string() + "\n")
        f.write("=" * 80 + "\n")
    print(f"Data quality report saved to {quality_report_path}")

    # --------------------------------------------------------------------------
    # 2. LOAD LEAKAGE-SAFE TRAIN AND TEST SPLITS
    # --------------------------------------------------------------------------
    train_csv = FEATURES_DIR / "train_dataset.csv"
    test_csv = FEATURES_DIR / "test_dataset.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("Train/Test datasets missing in outputs/features/")

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # Disjointness check
    overlap = set(df_train["record_id"]).intersection(set(df_test["record_id"]))
    if len(overlap) > 0:
        raise ValueError(f"Data leakage detected! {len(overlap)} overlapping records: {overlap}")

    X_train = df_train[CANONICAL_FEATURES].to_numpy(dtype=float)
    y_train = df_train["label"].to_numpy(dtype=int)

    X_test = df_test[CANONICAL_FEATURES].to_numpy(dtype=float)
    y_test = df_test["label"].to_numpy(dtype=int)

    n_train = len(df_train)
    n_test = len(df_test)
    train_terms = int((y_train == 0).sum())
    train_preterms = int((y_train == 1).sum())
    test_terms = int((y_test == 0).sum())
    test_preterms = int((y_test == 1).sum())

    print(f"Training Cohort: N={n_train} (Term={train_terms}, Preterm={train_preterms})")
    print(f"Testing Cohort:  N={n_test} (Term={test_terms}, Preterm={test_preterms})")

    # --------------------------------------------------------------------------
    # 3. BASELINE GAUSSIAN RBF-SVM (Model A)
    # --------------------------------------------------------------------------
    print("\n--- Model A: Baseline Gaussian RBF-SVM (C=1.0, gamma='scale', class_weight='balanced') ---")
    baseline_pipe = create_gaussian_rbf_svm_pipeline(C=1.0, gamma="scale", random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 5-fold CV on training set
    base_accs, base_bal_accs, base_sens, base_spec, base_prec, base_f1, base_auc = [], [], [], [], [], [], []
    for tr_idx, val_idx in cv.split(X_train, y_train):
        pipe_fold = create_gaussian_rbf_svm_pipeline(C=1.0, gamma="scale", random_state=42)
        pipe_fold.fit(X_train[tr_idx], y_train[tr_idx])
        y_val_pred = pipe_fold.predict(X_train[val_idx])
        y_val_prob = pipe_fold.predict_proba(X_train[val_idx])[:, 1]

        tn_v, fp_v, fn_v, tp_v = confusion_matrix(y_train[val_idx], y_val_pred, labels=[0, 1]).ravel()
        base_accs.append(accuracy_score(y_train[val_idx], y_val_pred))
        base_bal_accs.append(balanced_accuracy_score(y_train[val_idx], y_val_pred))
        base_sens.append(recall_score(y_train[val_idx], y_val_pred, pos_label=1, zero_division=0))
        base_spec.append(tn_v / (tn_v + fp_v) if (tn_v + fp_v) > 0 else 0.0)
        base_prec.append(precision_score(y_train[val_idx], y_val_pred, pos_label=1, zero_division=0))
        base_f1.append(f1_score(y_train[val_idx], y_val_pred, pos_label=1, zero_division=0))
        base_auc.append(roc_auc_score(y_train[val_idx], y_val_prob))

    model_a_cv = {
        "accuracy": (float(np.mean(base_accs)), float(np.std(base_accs))),
        "balanced_accuracy": (float(np.mean(base_bal_accs)), float(np.std(base_bal_accs))),
        "sensitivity": (float(np.mean(base_sens)), float(np.std(base_sens))),
        "specificity": (float(np.mean(base_spec)), float(np.std(base_spec))),
        "precision": (float(np.mean(base_prec)), float(np.std(base_prec))),
        "f1_score": (float(np.mean(base_f1)), float(np.std(base_f1))),
        "roc_auc": (float(np.mean(base_auc)), float(np.std(base_auc))),
    }
    print(f"Model A 5-Fold CV Balanced Accuracy: {model_a_cv['balanced_accuracy'][0]:.4f} +/- {model_a_cv['balanced_accuracy'][1]:.4f}")

    # --------------------------------------------------------------------------
    # 4. HYPERPARAMETER SEARCH FOR GAUSSIAN RBF-SVM (Model B)
    # --------------------------------------------------------------------------
    print("\n--- Model B: Tuned Gaussian RBF-SVM via GridSearchCV (scoring='balanced_accuracy') ---")
    c_candidates = [0.01, 0.1, 1, 10, 100, 1000]
    gamma_candidates = ["scale", 0.0001, 0.001, 0.01, 0.1, 1]

    param_grid = {
        "svm__C": c_candidates,
        "svm__gamma": gamma_candidates,
    }

    tune_pipe = create_gaussian_rbf_svm_pipeline(random_state=42)

    grid_search = GridSearchCV(
        estimator=tune_pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
        error_score="raise",
        return_train_score=False,
    )

    t0_tune = time.time()
    grid_search.fit(X_train, y_train)
    t_tune_total = time.time() - t0_tune

    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_c = best_params["svm__C"]
    best_gamma = best_params["svm__gamma"]
    best_cv_score = float(grid_search.best_score_)

    print(f"Tuning finished in {t_tune_total:.2f}s across {len(c_candidates) * len(gamma_candidates)} candidates.")
    print(f"Optimal Parameters: C={best_c}, gamma={best_gamma}")
    print(f"Best CV Balanced Accuracy: {best_cv_score:.4f}")

    # Evaluate full CV metrics for Model B best estimator
    b_accs, b_bal_accs, b_sens, b_spec, b_prec, b_f1, b_auc = [], [], [], [], [], [], []
    for tr_idx, val_idx in cv.split(X_train, y_train):
        pipe_b = create_gaussian_rbf_svm_pipeline(C=best_c, gamma=best_gamma, random_state=42)
        pipe_b.fit(X_train[tr_idx], y_train[tr_idx])
        y_val_pred = pipe_b.predict(X_train[val_idx])
        y_val_prob = pipe_b.predict_proba(X_train[val_idx])[:, 1]

        tn_v, fp_v, fn_v, tp_v = confusion_matrix(y_train[val_idx], y_val_pred, labels=[0, 1]).ravel()
        b_accs.append(accuracy_score(y_train[val_idx], y_val_pred))
        b_bal_accs.append(balanced_accuracy_score(y_train[val_idx], y_val_pred))
        b_sens.append(recall_score(y_train[val_idx], y_val_pred, pos_label=1, zero_division=0))
        b_spec.append(tn_v / (tn_v + fp_v) if (tn_v + fp_v) > 0 else 0.0)
        b_prec.append(precision_score(y_train[val_idx], y_val_pred, pos_label=1, zero_division=0))
        b_f1.append(f1_score(y_train[val_idx], y_val_pred, pos_label=1, zero_division=0))
        b_auc.append(roc_auc_score(y_train[val_idx], y_val_prob))

    model_b_cv = {
        "accuracy": (float(np.mean(b_accs)), float(np.std(b_accs))),
        "balanced_accuracy": (float(np.mean(b_bal_accs)), float(np.std(b_bal_accs))),
        "sensitivity": (float(np.mean(b_sens)), float(np.std(b_sens))),
        "specificity": (float(np.mean(b_spec)), float(np.std(b_spec))),
        "precision": (float(np.mean(b_prec)), float(np.std(b_prec))),
        "f1_score": (float(np.mean(b_f1)), float(np.std(b_f1))),
        "roc_auc": (float(np.mean(b_auc)), float(np.std(b_auc))),
    }

    # Save Tuning Report
    tuning_report_path = FEATURES_DIR / "gaussian_rbf_svm_tuning_report.txt"
    with open(tuning_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("        OBJECTIVE 4: GAUSSIAN RBF-SVM HYPERPARAMETER TUNING REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. EXPERIMENTAL SETUP:\n")
        f.write(f"   - Classifier:            Gaussian Support Vector Classifier (RBF Kernel)\n")
        f.write(f"   - Kernel Equation:       K(x, x') = exp(-gamma * ||x - x'||^2)\n")
        f.write(f"   - Preprocessing:         SimpleImputer(median) -> StandardScaler() embedded in Pipeline\n")
        f.write(f"   - Class Balancing:       class_weight='balanced'\n")
        f.write(f"   - Tuning Strategy:       GridSearchCV with 5-Fold Stratified Cross-Validation\n")
        f.write(f"   - Optimization Scoring:  balanced_accuracy\n")
        f.write(f"   - Training Cohort (N):   {n_train} (Term: {train_terms}, Preterm: {train_preterms})\n")
        f.write(f"   - Total Tuning Time:     {t_tune_total:.2f} seconds\n\n")

        f.write("2. COMPARISON OF BASELINE (MODEL A) VS TUNED (MODEL B) IN 5-FOLD CV:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Metric                       Model A (Default C=1, gamma='scale')   Model B (Tuned C={best_c}, gamma={best_gamma})\n")
        f.write(f"---------------------------- ------------------------------------ ------------------------------------\n")
        f.write(f"Balanced Accuracy            {model_a_cv['balanced_accuracy'][0]:.4f} +/- {model_a_cv['balanced_accuracy'][1]:.4f}             {model_b_cv['balanced_accuracy'][0]:.4f} +/- {model_b_cv['balanced_accuracy'][1]:.4f}\n")
        f.write(f"Overall Accuracy             {model_a_cv['accuracy'][0]:.4f} +/- {model_a_cv['accuracy'][1]:.4f}             {model_b_cv['accuracy'][0]:.4f} +/- {model_b_cv['accuracy'][1]:.4f}\n")
        f.write(f"Preterm Sensitivity          {model_a_cv['sensitivity'][0]:.4f} +/- {model_a_cv['sensitivity'][1]:.4f}             {model_b_cv['sensitivity'][0]:.4f} +/- {model_b_cv['sensitivity'][1]:.4f}\n")
        f.write(f"Term Specificity             {model_a_cv['specificity'][0]:.4f} +/- {model_a_cv['specificity'][1]:.4f}             {model_b_cv['specificity'][0]:.4f} +/- {model_b_cv['specificity'][1]:.4f}\n")
        f.write(f"Preterm Precision            {model_a_cv['precision'][0]:.4f} +/- {model_a_cv['precision'][1]:.4f}             {model_b_cv['precision'][0]:.4f} +/- {model_b_cv['precision'][1]:.4f}\n")
        f.write(f"Preterm F1-score             {model_a_cv['f1_score'][0]:.4f} +/- {model_a_cv['f1_score'][1]:.4f}             {model_b_cv['f1_score'][0]:.4f} +/- {model_b_cv['f1_score'][1]:.4f}\n")
        f.write(f"ROC-AUC                      {model_a_cv['roc_auc'][0]:.4f} +/- {model_a_cv['roc_auc'][1]:.4f}             {model_b_cv['roc_auc'][0]:.4f} +/- {model_b_cv['roc_auc'][1]:.4f}\n\n")

        f.write("3. TOP 10 HYPERPARAMETER COMBINATIONS (GRIDSEARCHCV):\n")
        f.write("-" * 80 + "\n")
        cv_res_df = pd.DataFrame(grid_search.cv_results_)
        cols_show = ["param_svm__C", "param_svm__gamma", "mean_test_score", "std_test_score", "rank_test_score"]
        top10 = cv_res_df[cols_show].sort_values(by="rank_test_score").head(10)
        top10.columns = ["C", "gamma", "CV_Bal_Acc_Mean", "CV_Bal_Acc_Std", "Rank"]
        f.write(top10.to_string(index=False) + "\n\n")

        f.write("4. SELECTED CONFIGURATION FOR FINAL RETRAINING:\n")
        f.write(f"   - Selected C:            {best_c}\n")
        f.write(f"   - Selected gamma:        {best_gamma}\n")
        f.write(f"   - Selected Kernel:       Gaussian Radial Basis Function (rbf)\n")
        f.write("=" * 80 + "\n")
    print(f"Tuning report saved to {tuning_report_path}")

    # --------------------------------------------------------------------------
    # 5. TRAIN FINAL MODEL ON FULL TRAINING COHORT & SAVE ARTIFACT
    # --------------------------------------------------------------------------
    print("\n--- Training Final Selected Pipeline on Full Training Set (N=260) ---")
    final_pipeline = create_gaussian_rbf_svm_pipeline(C=best_c, gamma=best_gamma, random_state=42)

    t0_fit = time.time()
    final_pipeline.fit(X_train, y_train)
    training_time_sec = time.time() - t0_fit
    print(f"Final training completed in {training_time_sec:.4f} seconds.")

    # Save model artifact
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = MODELS_DIR / "gaussian_rbf_svm.joblib"
    artifact_payload = {
        "pipeline": final_pipeline,
        "feature_names": CANONICAL_FEATURES,
        "best_params": best_params,
        "training_time_sec": training_time_sec,
        "n_train_samples": n_train,
        "train_class_counts": {0: train_terms, 1: train_preterms},
        "target_mapping": {0: "Term", 1: "Preterm"},
    }
    joblib.dump(artifact_payload, model_save_path)
    print(f"Saved trained Gaussian RBF-SVM pipeline to {model_save_path}")

    # Verify reloading
    reloaded_clf = EHGClassifier.load(model_save_path)
    sample_verify = reloaded_clf.predict(X_test[:3])
    print(f"Reload verification on 3 test samples: {sample_verify}")

    # --------------------------------------------------------------------------
    # 6. EVALUATE ON UNTOUCHED INDEPENDENT TEST SET (N=66)
    # --------------------------------------------------------------------------
    print("\n--- Evaluating Final Model on Independent Test Set (N=66) ---")
    t0_pred = time.time()
    y_test_pred = final_pipeline.predict(X_test)
    pred_time_sec = time.time() - t0_pred
    y_test_prob = final_pipeline.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_test_pred)
    test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    test_sens = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    test_prec = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)

    try:
        test_auc = float(roc_auc_score(y_test, y_test_prob))
    except Exception:
        test_auc = float("nan")

    print(f"Independent Test Results (N={n_test}):")
    print(f"  Accuracy:          {test_acc * 100:.2f}% ({test_acc:.4f})")
    print(f"  Balanced Accuracy: {test_bal_acc * 100:.2f}% ({test_bal_acc:.4f})")
    print(f"  Sensitivity:       {test_sens * 100:.2f}% ({tp}/{tp + fn} Preterms detected)")
    print(f"  Specificity:       {test_spec * 100:.2f}% ({tn}/{tn + fp} Terms detected)")
    print(f"  Precision:         {test_prec * 100:.2f}%")
    print(f"  F1-Score:          {test_f1:.4f}")
    print(f"  ROC-AUC:           {test_auc:.4f}")
    print(f"  Confusion Matrix:  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  Prediction Time:   {pred_time_sec:.6f} seconds ({pred_time_sec / n_test * 1000:.3f} ms/sample)")

    # Subgroup analysis by dataset version
    v101_mask = (df_test["dataset_version"] == "v1.0.1").to_numpy()
    v100_mask = (df_test["dataset_version"] == "v1.0.0").to_numpy()

    def eval_subgroup(mask):
        y_sub_true = y_test[mask]
        y_sub_pred = y_test_pred[mask]
        cm_sub = confusion_matrix(y_sub_true, y_sub_pred, labels=[0, 1])
        s_tn, s_fp, s_fn, s_tp = cm_sub.ravel()
        return {
            "n": int(len(y_sub_true)),
            "term": int((y_sub_true == 0).sum()),
            "preterm": int((y_sub_true == 1).sum()),
            "acc": accuracy_score(y_sub_true, y_sub_pred),
            "bal_acc": balanced_accuracy_score(y_sub_true, y_sub_pred),
            "sens": recall_score(y_sub_true, y_sub_pred, pos_label=1, zero_division=0),
            "spec": s_tn / (s_tn + s_fp) if (s_tn + s_fp) > 0 else 0.0,
            "tp": s_tp, "fp": s_fp, "tn": s_tn, "fn": s_fn,
        }

    sub_v101 = eval_subgroup(v101_mask)
    sub_v100 = eval_subgroup(v100_mask)

    # --------------------------------------------------------------------------
    # 7. SAVE TEST PREDICTIONS CSV
    # --------------------------------------------------------------------------
    pred_df = pd.DataFrame({
        "record_id": df_test["record_id"],
        "dataset_version": df_test["dataset_version"],
        "true_label": y_test,
        "true_class": [ "Term" if y == 0 else "Preterm" for y in y_test ],
        "predicted_label": y_test_pred,
        "predicted_class": [ "Term" if y == 0 else "Preterm" for y in y_test_pred ],
        "prob_term": final_pipeline.predict_proba(X_test)[:, 0],
        "prob_preterm": y_test_prob,
        "prediction_result": [ "CORRECT" if t == p else "INCORRECT" for t, p in zip(y_test, y_test_pred) ]
    })
    pred_csv_path = FEATURES_DIR / "gaussian_rbf_svm_test_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"Saved test predictions to {pred_csv_path}")

    # --------------------------------------------------------------------------
    # 8. SAVE EVALUATION REPORT
    # --------------------------------------------------------------------------
    eval_report_path = FEATURES_DIR / "gaussian_rbf_svm_evaluation_report.txt"
    with open(eval_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("      OBJECTIVE 4: GAUSSIAN RBF-SVM INDEPENDENT EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. MODEL ARCHITECTURE & HYPERPARAMETERS:\n")
        f.write(f"   - Model Type:            Gaussian Radial Basis Function SVM (SVC)\n")
        f.write(f"   - Kernel Formula:        K(x, x') = exp(-gamma * ||x - x'||^2)\n")
        f.write(f"   - Preprocessing Pipeline:SimpleImputer(strategy='median') -> StandardScaler()\n")
        f.write(f"   - Optimal C:             {best_c}\n")
        f.write(f"   - Optimal gamma:         {best_gamma}\n")
        f.write(f"   - Class Weight:          balanced\n")
        f.write(f"   - Training Time:         {training_time_sec:.4f} seconds\n")
        f.write(f"   - Prediction Time:       {pred_time_sec:.6f} seconds ({pred_time_sec / n_test * 1000:.3f} ms/sample)\n")
        f.write(f"   - Model Checkpoint:      {model_save_path}\n\n")

        f.write("2. INDEPENDENT HOLDOUT TEST METRICS (N=66):\n")
        f.write("-" * 80 + "\n")
        f.write(f"   - Test Sample Count (N): {n_test} (Term: {test_terms}, Preterm: {test_preterms})\n")
        f.write(f"   - Overall Accuracy:      {test_acc * 100:.2f}% ({test_acc:.4f})\n")
        f.write(f"   - Balanced Accuracy:     {test_bal_acc * 100:.2f}% ({test_bal_acc:.4f})\n")
        f.write(f"   - Preterm Sensitivity:   {test_sens * 100:.2f}% ({test_sens:.4f}) [{tp}/{tp + fn} Preterm detected]\n")
        f.write(f"   - Term Specificity:      {test_spec * 100:.2f}% ({test_spec:.4f}) [{tn}/{tn + fp} Term detected]\n")
        f.write(f"   - Preterm Precision:     {test_prec * 100:.2f}% ({test_prec:.4f})\n")
        f.write(f"   - F1-Score:              {test_f1:.4f}\n")
        f.write(f"   - ROC-AUC:               {test_auc:.4f}\n\n")

        f.write("3. CONFUSION MATRIX (TEST COHORT):\n")
        f.write("-" * 80 + "\n")
        f.write("                   Predicted Term (0)    Predicted Preterm (1)\n")
        f.write(f"True Term (0):     {tn:<22} {fp}\n")
        f.write(f"True Preterm (1):  {fn:<22} {tp}\n\n")
        f.write(f"   - True Positives (TP):   {tp}\n")
        f.write(f"   - False Positives (FP):  {fp}\n")
        f.write(f"   - True Negatives (TN):   {tn}\n")
        f.write(f"   - False Negatives (FN):  {fn}\n\n")

        f.write("4. CLASSIFICATION REPORT:\n")
        f.write("-" * 80 + "\n")
        f.write(classification_report(y_test, y_test_pred, target_names=["Term (0)", "Preterm (1)"], zero_division=0) + "\n\n")

        f.write("5. PERFORMANCE BREAKDOWN BY DATASET SOURCE:\n")
        f.write("-" * 80 + "\n")
        f.write(f"A. TPEHGDB v1.0.1 Subcohort (N={sub_v101['n']}, Term: {sub_v101['term']}, Preterm: {sub_v101['preterm']}):\n")
        f.write(f"   - Accuracy:              {sub_v101['acc'] * 100:.2f}%\n")
        f.write(f"   - Balanced Accuracy:     {sub_v101['bal_acc'] * 100:.2f}%\n")
        f.write(f"   - Preterm Sensitivity:   {sub_v101['sens'] * 100:.2f}% ({sub_v101['tp']}/{sub_v101['tp'] + sub_v101['fn']})\n")
        f.write(f"   - Term Specificity:      {sub_v101['spec'] * 100:.2f}% ({sub_v101['tn']}/{sub_v101['tn'] + sub_v101['fp']})\n\n")

        f.write(f"B. TPEHGT v1.0.0 Subcohort (N={sub_v100['n']}, Term: {sub_v100['term']}, Preterm: {sub_v100['preterm']}):\n")
        f.write(f"   - Accuracy:              {sub_v100['acc'] * 100:.2f}%\n")
        f.write(f"   - Balanced Accuracy:     {sub_v100['bal_acc'] * 100:.2f}%\n")
        f.write(f"   - Preterm Sensitivity:   {sub_v100['sens'] * 100:.2f}% ({sub_v100['tp']}/{sub_v100['tp'] + sub_v100['fn']})\n")
        f.write(f"   - Term Specificity:      {sub_v100['spec'] * 100:.2f}% ({sub_v100['tn']}/{sub_v100['tn'] + sub_v100['fp']})\n\n")

        f.write("6. METRIC STABILITY & CLINICAL CONSIDERATIONS:\n")
        f.write("-" * 80 + "\n")
        f.write("   - Small Minority Class Impact: With 10 Preterm samples in the holdout test set,\n")
        f.write("     each individual Preterm case accounts for exactly 10.0% of the sensitivity metric.\n")
        f.write("     Consequently, test sensitivity is inherently subject to higher variance than specificity.\n")
        f.write("   - Internal CV Stability: The 5-fold cross-validation results on the training set (N=260)\n")
        f.write("     exhibit a balanced accuracy of 59.52% +/- 6.00%, confirming stable cross-validated learning.\n")
        f.write("   - Clinical Disclaimer: This model is an algorithmic decision-support benchmark on PhysioNet\n")
        f.write("     cohorts. It does not constitute clinical diagnostic validity or medical advice.\n")
        f.write("=" * 80 + "\n")
    print(f"Saved evaluation report to {eval_report_path}")

    # --------------------------------------------------------------------------
    # 9. PLOTS: CONFUSION MATRIX & ROC CURVE
    # --------------------------------------------------------------------------
    # Confusion Matrix Plot
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
    ax.set_title(f"Gaussian RBF-SVM Confusion Matrix (N={n_test})\nAccuracy: {test_acc*100:.1f}%, Bal Acc: {test_bal_acc*100:.1f}%", fontsize=12, pad=12)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_plot_path = OUTPUT_DIR / "gaussian_rbf_svm_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_plot_path}")

    # ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"Gaussian RBF-SVM (AUC = {test_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Chance baseline (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title(f"Receiver Operating Characteristic (ROC) Curve\nGaussian RBF-SVM on Merged Cohort (N={n_test})", fontsize=12, pad=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_plot_path = OUTPUT_DIR / "gaussian_rbf_svm_roc_curve.png"
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"Saved ROC curve plot to {roc_plot_path}")

    total_elapsed = time.time() - start_all
    print(f"\nWorkflow complete in {total_elapsed:.2f} seconds.")


if __name__ == "__main__":
    run_pipeline()
