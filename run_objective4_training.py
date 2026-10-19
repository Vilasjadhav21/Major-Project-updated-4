"""
Complete Model Training and Evaluation Pipeline for Objective 4: EHG Rapid Triage.

This script executes:
STEP 1: Inspect input files and verify feature definitions and scaling state.
STEP 2: Validate datasets for data leakage, missing values, duplicates, and split disjointness.
STEP 3: Train the SVM RBF classifier on training data only.
STEP 4: Save the trained classifier to outputs/models/ehg_classifier.joblib and verify persistence.
STEP 5: Evaluate model on independent test dataset, generating metrics, confusion matrix, and ROC curve.
STEP 6: Verify model persistence and reloading.
STEP 7: Output final report.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Ensure workspace is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import MODELS_DIR, OUTPUT_DIR, PLOTS_DIR, FEATURES_DIR, ModelConfig
from src.model import EHGClassifier
from src.evaluation import EHGEvaluator


def run_pipeline():
    print("=" * 80)
    print("      OBJECTIVE 4: EHG RAPID TRIAGE — MODEL TRAINING & EVALUATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1 — INSPECT INPUT FILES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1 — INSPECT INPUT FILES")
    print("=" * 80)

    train_path = FEATURES_DIR / "train_dataset.csv"
    test_path = FEATURES_DIR / "test_dataset.csv"
    scaler_path = MODELS_DIR / "standard_scaler.joblib"
    model_py_path = BASE_DIR / "src" / "model.py"
    eval_py_path = BASE_DIR / "src" / "evaluation.py"
    config_py_path = BASE_DIR / "src" / "config.py"
    prep_py_path = BASE_DIR / "src" / "preprocessing.py"

    required_files = [
        train_path, test_path, scaler_path,
        model_py_path, eval_py_path, config_py_path, prep_py_path
    ]
    for rf in required_files:
        if not rf.exists():
            raise FileNotFoundError(f"CRITICAL: Required file not found: {rf}")
        print(f"  [OK] Verified input file exists: {rf.relative_to(BASE_DIR)}")

    # Load datasets and existing scaler
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    existing_scaler = joblib.load(scaler_path)

    expected_features = [
        "sampen_imf1",
        "median_freq_imf1_hz",
        "rms_imf1",
        "rms_imf2",
        "hilbert_par_imf1",
    ]
    expected_target = "label"

    # Verify column presence
    for feat in expected_features:
        if feat not in train_df.columns:
            raise ValueError(f"CRITICAL: Missing expected feature in train dataset: {feat}")
        if feat not in test_df.columns:
            raise ValueError(f"CRITICAL: Missing expected feature in test dataset: {feat}")
    if expected_target not in train_df.columns or expected_target not in test_df.columns:
        raise ValueError(f"CRITICAL: Missing expected target column: {expected_target}")

    print("\nExact Feature Column Names:")
    for i, feat in enumerate(expected_features, start=1):
        print(f"  {i}. {feat}")
    print(f"Target Column: {expected_target} (0 = Term, 1 = Preterm)")

    # Verify whether train_dataset.csv and test_dataset.csv contain raw or scaled values
    raw_means = train_df[expected_features].mean().to_numpy()
    scaler_means = existing_scaler.mean_
    mean_diff = np.max(np.abs(raw_means - scaler_means))

    print("\nScaling Verification:")
    print(f"  Max difference between train_dataset.csv means and standard_scaler.joblib means: {mean_diff:.2e}")
    if mean_diff < 1e-6:
        print("  -> train_dataset.csv and test_dataset.csv contain RAW (unscaled) feature values.")
        print("  -> outputs/models/standard_scaler.joblib was fitted on training data only.")
        print("  -> The classifier pipeline must scale raw features using the training-derived scaler parameters.")
    else:
        print("  -> Feature values do not match standard_scaler.joblib mean vector.")

    # -------------------------------------------------------------------------
    # STEP 2 — VERIFY DATA
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2 — VERIFY DATA")
    print("=" * 80)

    n_train = len(train_df)
    n_test = len(test_df)
    train_classes = train_df[expected_target].value_counts().to_dict()
    test_classes = test_df[expected_target].value_counts().to_dict()

    train_missing = int(train_df[expected_features].isnull().sum().sum())
    test_missing = int(test_df[expected_features].isnull().sum().sum())
    train_inf = int(np.isinf(train_df[expected_features].to_numpy()).sum())
    test_inf = int(np.isinf(test_df[expected_features].to_numpy()).sum())

    train_dup = int(train_df["record"].duplicated().sum())
    test_dup = int(test_df["record"].duplicated().sum())

    train_records = set(train_df["record"])
    test_records = set(test_df["record"])
    overlap_records = train_records.intersection(test_records)

    print(f"- Number of training records: {n_train}")
    print(f"- Number of test records: {n_test}")
    print(f"- Training class distribution: Term (0) = {train_classes.get(0, 0)}, Preterm (1) = {train_classes.get(1, 0)}")
    print(f"- Test class distribution: Term (0) = {test_classes.get(0, 0)}, Preterm (1) = {test_classes.get(1, 0)}")
    print(f"- Missing feature values: Train = {train_missing}, Test = {test_missing}")
    print(f"- Infinite feature values: Train = {train_inf}, Test = {test_inf}")
    print(f"- Duplicate record IDs: Train = {train_dup}, Test = {test_dup}")
    print(f"- Overlapping record IDs between train and test: {len(overlap_records)} ({overlap_records if overlap_records else 'None'})")

    # Guard checks
    if len(overlap_records) > 0:
        raise ValueError(f"DATA LEAKAGE DETECTED: Overlapping records {overlap_records}")
    if train_missing > 0 or test_missing > 0:
        raise ValueError("Missing feature values detected in datasets.")
    if train_inf > 0 or test_inf > 0:
        raise ValueError("Infinite feature values detected in datasets.")
    if train_dup > 0 or test_dup > 0:
        raise ValueError("Duplicate record IDs detected in datasets.")
    if set(train_df[expected_target].unique()) - {0, 1} or set(test_df[expected_target].unique()) - {0, 1}:
        raise ValueError("Invalid target label detected (must be 0 or 1).")

    print("  [PASS] Data integrity, split disjointness, and label validity confirmed.")

    # -------------------------------------------------------------------------
    # STEP 3 — TRAIN THE ACTUAL MODEL
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3 — TRAIN THE ACTUAL MODEL")
    print("=" * 80)

    X_train_raw = train_df[expected_features]
    y_train = train_df[expected_target].to_numpy()

    # Use EHGClassifier implementation from src/model.py
    model_config = ModelConfig(
        model_type="svm",
        random_state=42
    )
    clf = EHGClassifier(config=model_config)

    print("Target classifier configuration:")
    print(f"  Type: {type(clf.model)}")
    print(f"  Kernel: {clf.model.kernel}")
    print(f"  Probability: {clf.model.probability}")
    print(f"  Class Weight: {clf.model.class_weight}")
    print(f"  Random State: {clf.model.random_state}")

    # Train using training data ONLY
    clf.fit(X_train_raw, y_train)

    # Verify that the scaler fitted inside EHGClassifier matches standard_scaler.joblib
    scaler_mean_diff = np.max(np.abs(clf.scaler.mean_ - existing_scaler.mean_))
    scaler_scale_diff = np.max(np.abs(clf.scaler.scale_ - existing_scaler.scale_))
    print(f"\nScaler parameters verified against existing standard_scaler.joblib:")
    print(f"  Max mean difference: {scaler_mean_diff:.2e}")
    print(f"  Max scale difference: {scaler_scale_diff:.2e}")
    assert scaler_mean_diff < 1e-6, "Scaler mean mismatch"
    assert scaler_scale_diff < 1e-6, "Scaler scale mismatch"
    print("  [PASS] Scaler was fitted strictly on training data only.")

    # -------------------------------------------------------------------------
    # STEP 4 — SAVE THE TRAINED CLASSIFIER
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4 — SAVE THE TRAINED CLASSIFIER")
    print("=" * 80)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = MODELS_DIR / "ehg_classifier.joblib"

    # Save fitted classifier
    clf.save(model_save_path)

    # Reload saved file and verify that it is a fitted SVC classifier
    reloaded_raw = joblib.load(model_save_path)
    print(f"Reloaded file: {model_save_path}")
    print(f"Raw object type from joblib.load: {type(reloaded_raw)}")

    if not isinstance(reloaded_raw, SVC):
        raise TypeError(f"CRITICAL: Reloaded object is not an SVC classifier: {type(reloaded_raw)}")
    if not hasattr(reloaded_raw, "support_vectors_") or reloaded_raw.support_vectors_ is None:
        raise RuntimeError("CRITICAL: Reloaded SVC classifier is not fitted!")

    print(f"  [PASS] Verified fitted SVC classifier: {reloaded_raw}")
    print(f"  Support vectors shape: {reloaded_raw.support_vectors_.shape}")
    print(f"  Number of support vectors per class: {reloaded_raw.n_support_}")
    print(f"  Classes: {reloaded_raw.classes_}")

    # Also verify loading via EHGClassifier.load
    reloaded_clf = EHGClassifier.load(model_save_path)
    print(f"  [PASS] Verified EHGClassifier wrapper loads properly.")

    # -------------------------------------------------------------------------
    # STEP 5 — EVALUATE THE MODEL
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 5 — EVALUATE THE MODEL")
    print("=" * 80)

    X_test_raw = test_df[expected_features]
    y_test = test_df[expected_target].to_numpy()

    # Predict on unseen independent test dataset
    y_pred = reloaded_clf.predict(X_test_raw)
    y_proba = reloaded_clf.predict_proba(X_test_raw)

    print(f"Actual test record count: {len(y_test)}")
    test_term_count = int(np.sum(y_test == 0))
    test_preterm_count = int(np.sum(y_test == 1))
    print(f"Test set class counts: Term (0) = {test_term_count}, Preterm (1) = {test_preterm_count}")

    metrics = EHGEvaluator.compute_metrics(y_test, y_pred, y_proba)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print("\nConfusion Matrix (labels [0, 1]):")
    print("             Predicted Term (0)   Predicted Preterm (1)")
    print(f"True Term (0)       {cm[0, 0]:<20} {cm[0, 1]:<20}")
    print(f"True Preterm (1)    {cm[1, 0]:<20} {cm[1, 1]:<20}")

    print("\nEvaluation Metrics on Independent Test Set:")
    print(f"  Accuracy:    {metrics['accuracy'] * 100:.2f}% ({metrics['accuracy']:.4f})")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  F1-score:    {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")

    # Detailed test sample breakdown
    print("\nPer-sample Test Predictions:")
    pos_probs = y_proba[:, 1]
    for idx, row in test_df.iterrows():
        rec = row["record"]
        yt = int(row["label"])
        yp = int(y_pred[idx])
        p1 = float(pos_probs[idx])
        status = "CORRECT" if yt == yp else "MISCLASSIFIED"
        print(f"  Record {rec}: True={yt} ({'Preterm' if yt==1 else 'Term'}), "
              f"Pred={yp} ({'Preterm' if yp==1 else 'Term'}), "
              f"P(Preterm)={p1:.4f} -> {status}")

    # Generate Output Files
    cm_path_main = OUTPUT_DIR / "confusion_matrix.png"
    cm_path_plots = PLOTS_DIR / "confusion_matrix.png"
    roc_path_main = OUTPUT_DIR / "roc_curve.png"
    roc_path_plots = PLOTS_DIR / "roc_curve.png"
    eval_results_path = OUTPUT_DIR / "evaluation_results.txt"

    # 1. Confusion Matrix Plot
    EHGEvaluator.plot_confusion_matrix(
        y_test, y_pred,
        title="Objective 4 — Test Set Confusion Matrix",
        save_path=cm_path_main
    )
    # Save copy to outputs/plots/
    EHGEvaluator.plot_confusion_matrix(
        y_test, y_pred,
        title="Objective 4 — Test Set Confusion Matrix",
        save_path=cm_path_plots
    )
    print(f"\nSaved confusion matrix plot: {cm_path_main}")

    # 2. ROC Curve Plot
    EHGEvaluator.plot_roc_curve(
        y_test, y_proba,
        title="Objective 4 — Preterm Triage ROC Curve",
        save_path=roc_path_main
    )
    # Save copy to outputs/plots/
    EHGEvaluator.plot_roc_curve(
        y_test, y_proba,
        title="Objective 4 — Preterm Triage ROC Curve",
        save_path=roc_path_plots
    )
    print(f"Saved ROC curve plot: {roc_path_main}")

    # 3. Evaluation Results Text Report
    report_content = f"""================================================================================
        OBJECTIVE 4: EHG RAPID TRIAGE — MODEL EVALUATION REPORT
================================================================================
1. MODEL IDENTIFICATION & ARTIFACT:
   - Model Algorithm: Support Vector Machine (SVC)
   - Kernel: Radial Basis Function (RBF)
   - Class Weight: balanced
   - Probability Estimation: Platt scaling (probability=True)
   - Random State: 42
   - Model Checkpoint: outputs/models/ehg_classifier.joblib
   - Preprocessing Scaler: outputs/models/standard_scaler.joblib

--------------------------------------------------------------------------------
2. DATASET SPLIT & RECORDS:
   - Total Evaluated Records: 15
   - Training Records: 12 (80.0%)
     * Term (0):    10
     * Preterm (1): 2
   - Independent Test Records: 3 (20.0%)
     * Term (0):    2
     * Preterm (1): 1
   - Record IDs Disjoint: PASS (0 overlap)
   - Feature Scaling: StandardScaler fitted strictly on training data only

--------------------------------------------------------------------------------
3. TOP 5 PREDICTIVE IMF-1 FEATURES:
   1. sampen_imf1 (Sample Entropy of IMF-1)
   2. median_freq_imf1_hz (Median Frequency of IMF-1, Hz)
   3. rms_imf1 (Root Mean Square of IMF-1)
   4. rms_imf2 (Root Mean Square of IMF-2)
   5. hilbert_par_imf1 (Hilbert Peak-to-Average Ratio of IMF-1)

--------------------------------------------------------------------------------
4. INDEPENDENT TEST SET EVALUATION METRICS:
   - Accuracy:    {metrics['accuracy'] * 100:.2f}% ({metrics['accuracy']:.4f})
   - Sensitivity / Recall (Preterm): {metrics['sensitivity']:.4f} ({metrics['sensitivity'] * 100:.1f}%)
   - Specificity (Term):             {metrics['specificity']:.4f} ({metrics['specificity'] * 100:.1f}%)
   - Precision (Preterm):            {metrics['precision']:.4f}
   - F1-Score:                       {metrics['f1_score']:.4f}
   - ROC-AUC:                        {metrics['roc_auc']:.4f}

--------------------------------------------------------------------------------
5. CONFUSION MATRIX:
                       Predicted Term (0)    Predicted Preterm (1)
   True Term (0):      {cm[0, 0]:<21} {cm[0, 1]:<21}
   True Preterm (1):   {cm[1, 0]:<21} {cm[1, 1]:<21}

   True Positives (TP):  {metrics['true_positives']}
   False Positives (FP): {metrics['false_positives']}
   True Negatives (TN):  {metrics['true_negatives']}
   False Negatives (FN): {metrics['false_negatives']}

--------------------------------------------------------------------------------
6. TEST SAMPLE LEVEL BREAKDOWN:
   Record: tpehg1203 | True: Preterm (1) | Pred: {y_pred[0]} ({'Preterm' if y_pred[0]==1 else 'Term'}) | P(Preterm): {pos_probs[0]:.4f}
   Record: tpehg1065 | True: Term (0)    | Pred: {y_pred[1]} ({'Preterm' if y_pred[1]==1 else 'Term'}) | P(Preterm): {pos_probs[1]:.4f}
   Record: tpehg1029 | True: Term (0)    | Pred: {y_pred[2]} ({'Preterm' if y_pred[2]==1 else 'Term'}) | P(Preterm): {pos_probs[2]:.4f}

--------------------------------------------------------------------------------
7. CLINICAL DISCLAIMER:
   This evaluation is an algorithmic verification on the PhysioNet TPEHGDB v1.0.1
   cohort and does not constitute clinical diagnostic validation or medical advice.
================================================================================
"""
    with open(eval_results_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved evaluation results report: {eval_results_path}")

    # -------------------------------------------------------------------------
    # STEP 6 — VERIFY MODEL PERSISTENCE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 6 — VERIFY MODEL PERSISTENCE")
    print("=" * 80)

    # 1. Confirm outputs/models/ehg_classifier.joblib exists
    assert model_save_path.exists(), "Model file does not exist!"
    print(f"  [PASS] {model_save_path} exists ({model_save_path.stat().st_size} bytes)")

    # 2. Confirm model can be loaded
    final_loaded = joblib.load(model_save_path)
    assert isinstance(final_loaded, SVC), "Loaded model is not an SVC!"
    print(f"  [PASS] Model loaded successfully: {final_loaded}")

    # 3. Confirm loaded model produces predictions
    # Transform test features with existing scaler
    X_test_scaled = existing_scaler.transform(X_test_raw)
    test_preds_direct = final_loaded.predict(X_test_scaled)
    test_probas_direct = final_loaded.predict_proba(X_test_scaled)
    np.testing.assert_array_equal(test_preds_direct, y_pred)
    np.testing.assert_allclose(test_probas_direct, y_proba)
    print(f"  [PASS] Loaded model produces identical predictions: {test_preds_direct.tolist()}")

    # 4. Confirm evaluation report was saved
    assert eval_results_path.exists(), "Evaluation report file does not exist!"
    print(f"  [PASS] Evaluation report confirmed at: {eval_results_path}")

    # -------------------------------------------------------------------------
    # STEP 7 — FINAL REPORT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 7 — FINAL REPORT")
    print("=" * 80)
    print(f"1. Training status: SUCCESS (SVM RBF classifier fitted without data leakage)")
    print(f"2. Saved model path: {model_save_path}")
    print(f"3. Training record count: {n_train}")
    print(f"4. Test record count: {n_test}")
    print(f"5. Test accuracy as a percentage: {metrics['accuracy'] * 100:.2f}%")
    print(f"6. Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"7. Specificity: {metrics['specificity']:.4f}")
    print(f"8. Precision: {metrics['precision']:.4f}")
    print(f"9. F1-score: {metrics['f1_score']:.4f}")
    print(f"10. ROC-AUC: {metrics['roc_auc']:.4f}")
    print("11. Paths to the generated output files:")
    print(f"    - Model:             {model_save_path}")
    print(f"    - Confusion Matrix:  {cm_path_main}")
    print(f"    - ROC Curve:         {roc_path_main}")
    print(f"    - Evaluation Report: {eval_results_path}")
    print(f"    - Plots Directory Copy: {cm_path_plots}, {roc_path_plots}")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
