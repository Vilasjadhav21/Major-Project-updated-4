"""
Machine Learning Classifier Module for EHG Rapid Triage.

Implements a reproducible Gaussian RBF-SVM classification pipeline:
    SimpleImputer(strategy="median") -> StandardScaler() -> SVC(kernel="rbf", class_weight="balanced")

Mathematical Foundation:
    The RBF kernel is the Gaussian kernel:
    K(x, x') = exp(-gamma * ||x - x'||^2)

Preprocessing (imputation and scaling) is embedded inside the scikit-learn Pipeline
to guarantee that all transformations are fitted strictly within training folds,
eliminating data leakage during cross-validation and independent evaluation.

Features:
    1. Gaussian RBF-SVM Pipeline with median imputation and standardization
    2. Input validation for the five canonical EHG features
    3. Infinite and missing value detection and safe median imputation
    4. Stratified K-Fold Cross-Validation with clinical metrics
    5. Hyperparameter tuning (GridSearchCV) optimized for Balanced Accuracy
    6. Pipeline serialization and persistence
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate as sklearn_cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import ModelConfig, MODELS_DIR

ArrayInput = Union[pd.DataFrame, np.ndarray]

# 5 Canonical EHG features defined for Objective 4 Rapid Triage
CANONICAL_FEATURES: List[str] = [
    "sampen_imf1",
    "median_freq_imf1_hz",
    "rms_imf1",
    "rms_imf2",
    "hilbert_par_imf1",
]

# Metadata/provenance/target columns that must NEVER be passed to classifiers as features
NON_FEATURE_COLUMNS: List[str] = [
    "record",
    "record_id",
    "composite_id",
    "dataset_version",
    "label",
    "target",
    "group",
    "early",
    "gestation",
    "rec_time",
    "strata",
]


def create_gaussian_rbf_svm_pipeline(
    C: float = 1.0,
    gamma: Union[str, float] = "scale",
    random_state: int = 42,
    class_weight: str = "balanced",
) -> Pipeline:
    """
    Factory function for a reproducible Gaussian RBF-SVM pipeline.

    The RBF kernel is the Gaussian kernel:
        K(x, x') = exp(-gamma * ||x - x'||^2)

    Architecture:
        1. SimpleImputer: Replaces missing/inf values with median computed on training data.
        2. StandardScaler: Normalizes features to zero mean and unit variance.
        3. SVC: Gaussian RBF Support Vector Classifier with balanced class weights.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=C,
                    gamma=gamma,
                    kernel="rbf",
                    class_weight=class_weight,
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )


class EHGClassifier:
    """
    Machine learning classifier for EHG rapid triage.

    Default model: Gaussian RBF-SVM with embedded imputation and scaling.
    Also supports legacy model types (random_forest, gradient_boosting, logistic).
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        C: float = 1.0,
        gamma: Union[str, float] = "scale",
    ):
        """
        Initialize the classifier.

        Parameters
        ----------
        config : Optional[ModelConfig]
            Configuration object containing model settings.
        C : float, default=1.0
            Regularization parameter for SVM.
        gamma : str or float, default="scale"
            Kernel coefficient for Gaussian RBF kernel.
        """
        self.config = config or ModelConfig()
        self.C = getattr(self.config, "svm_c", C) or C
        self.gamma = gamma

        # Build pipeline
        self.pipeline: Pipeline = self._build_pipeline(self.config.model_type)

        self.best_params_: Optional[Dict] = None
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted_: bool = False
        self.training_time_sec_: Optional[float] = None
        self.invalid_values_detected_: int = 0

    @property
    def model(self):
        """Access underlying estimator step."""
        if "svm" in self.pipeline.named_steps:
            return self.pipeline.named_steps["svm"]
        elif "model" in self.pipeline.named_steps:
            return self.pipeline.named_steps["model"]
        return None

    @model.setter
    def model(self, estimator):
        if "svm" in self.pipeline.named_steps:
            self.pipeline.named_steps["svm"] = estimator
        elif "model" in self.pipeline.named_steps:
            self.pipeline.named_steps["model"] = estimator

    @property
    def scaler(self):
        """Access underlying scaler step."""
        if "scaler" in self.pipeline.named_steps:
            return self.pipeline.named_steps["scaler"]
        return None

    @scaler.setter
    def scaler(self, scaler_obj):
        if "scaler" in self.pipeline.named_steps:
            self.pipeline.named_steps["scaler"] = scaler_obj

    @property
    def imputer(self):
        """Access underlying imputer step."""
        if "imputer" in self.pipeline.named_steps:
            return self.pipeline.named_steps["imputer"]
        return None

    # ---------------------------------------------------------
    # 1. BUILD PIPELINE
    # ---------------------------------------------------------

    def _build_pipeline(self, model_type: str) -> Pipeline:
        """Create the appropriate scikit-learn Pipeline based on model_type."""
        model_type = model_type.lower().strip()

        if model_type in ("svm", "rbf_svm", "gaussian_rbf_svm"):
            return create_gaussian_rbf_svm_pipeline(
                C=self.C,
                gamma=self.gamma,
                random_state=self.config.random_state,
                class_weight="balanced",
            )
        elif model_type == "random_forest":
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=100,
                            random_state=self.config.random_state,
                            class_weight="balanced",
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
        elif model_type == "gradient_boosting":
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        GradientBoostingClassifier(
                            n_estimators=100,
                            random_state=self.config.random_state,
                        ),
                    ),
                ]
            )
        elif model_type == "logistic":
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=self.config.random_state,
                            class_weight="balanced",
                        ),
                    ),
                ]
            )
        else:
            raise ValueError(
                f"Unsupported model_type: {model_type}. Supported: svm, random_forest, gradient_boosting, logistic."
            )

    # ---------------------------------------------------------
    # 2. VALIDATE INPUT DATA
    # ---------------------------------------------------------

    def _validate_X(
        self, X: ArrayInput, require_canonical: bool = False
    ) -> Tuple[np.ndarray, Optional[List[str]]]:
        """
        Validate feature inputs, strip metadata/label columns, check canonical features,
        and safely sanitize NaN and infinite values for pipeline imputation.
        """
        feature_names: Optional[List[str]] = None
        invalid_count = 0

        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise ValueError("Input feature DataFrame is empty.")

            # 1. Never pass metadata, record IDs, or labels to the classifier
            clean_df = X.drop(
                columns=[col for col in NON_FEATURE_COLUMNS if col in X.columns],
                errors="ignore",
            )

            # 2. If canonical features requested or detected, strictly enforce their presence and order
            has_any_canonical = any(c in clean_df.columns for c in CANONICAL_FEATURES)
            if require_canonical or has_any_canonical:
                missing = [c for c in CANONICAL_FEATURES if c not in clean_df.columns]
                if missing:
                    raise ValueError(
                        f"Missing required canonical feature(s): {missing}. Expected all 5: {CANONICAL_FEATURES}"
                    )
                clean_df = clean_df[CANONICAL_FEATURES]
                feature_names = CANONICAL_FEATURES
            else:
                feature_names = list(clean_df.columns)

            # 3. Check for infinite or NaN values
            raw_vals = clean_df.to_numpy(dtype=float)
            is_nan = np.isnan(raw_vals)
            is_inf = np.isinf(raw_vals)
            invalid_count = int(np.sum(is_nan | is_inf))
            if invalid_count > 0:
                # Replace inf with NaN so SimpleImputer can impute median safely
                raw_vals[is_inf] = np.nan

            X_array = raw_vals

        else:
            X_array = np.asarray(X, dtype=float).copy()
            if X_array.ndim != 2:
                raise ValueError("X must be a 2D array of shape (samples, features).")
            if X_array.shape[0] == 0 or X_array.shape[1] == 0:
                raise ValueError("X contains no samples or features.")

            is_inf = np.isinf(X_array)
            invalid_count = int(np.sum(is_inf | np.isnan(X_array)))
            if invalid_count > 0:
                X_array[is_inf] = np.nan

        self.invalid_values_detected_ = invalid_count
        return X_array, feature_names

    @staticmethod
    def _validate_y(y: np.ndarray, n_samples: int) -> np.ndarray:
        """Validate target labels: must match n_samples, contain no NaNs, and have >= 2 classes."""
        y_array = np.asarray(y).ravel()

        if len(y_array) != n_samples:
            raise ValueError(
                f"Number of target labels ({len(y_array)}) does not match samples ({n_samples})."
            )

        if pd.isna(y_array).any():
            raise ValueError("Target labels contain missing/NaN values.")

        unique_classes = np.unique(y_array)
        if len(unique_classes) < 2:
            raise ValueError("At least two classes are required for classification.")

        # Ensure binary labels are standard 0 and 1
        if not set(unique_classes).issubset({0, 1}):
            raise ValueError(
                f"Binary classification targets must be 0 (Term) and 1 (Preterm), got {unique_classes}"
            )

        return y_array

    # ---------------------------------------------------------
    # 3. FIT PIPELINE
    # ---------------------------------------------------------

    def fit(self, X: ArrayInput, y: np.ndarray) -> "EHGClassifier":
        """
        Fit the Gaussian RBF-SVM pipeline (Imputer -> Scaler -> SVC).

        Preprocessing is fitted strictly on X using training data only.
        """
        X_array, feature_names = self._validate_X(X)
        y_array = self._validate_y(y, X_array.shape[0])

        self.feature_names_ = feature_names

        start_time = time.time()
        self.pipeline.fit(X_array, y_array)
        self.training_time_sec_ = time.time() - start_time

        self.is_fitted_ = True
        return self

    # ---------------------------------------------------------
    # 4. PREDICTION
    # ---------------------------------------------------------

    def predict(self, X: ArrayInput) -> np.ndarray:
        """Predict class labels (0 = Term, 1 = Preterm)."""
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        X_array, _ = self._validate_X(X)
        return self.pipeline.predict(X_array)

    def predict_proba(self, X: ArrayInput) -> np.ndarray:
        """Return class probability matrix with shape (n_samples, 2)."""
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        X_array, _ = self._validate_X(X)
        return self.pipeline.predict_proba(X_array)

    # ---------------------------------------------------------
    # 5. CROSS-VALIDATION
    # ---------------------------------------------------------

    def cross_validate(
        self,
        X: ArrayInput,
        y: np.ndarray,
        n_splits: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform Stratified K-Fold cross-validation on the pipeline.
        Imputation and scaling occur strictly inside each fold.
        """
        X_array, _ = self._validate_X(X)
        y_array = self._validate_y(y, X_array.shape[0])

        n_splits = n_splits or self.config.n_splits
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )

        accs, bal_accs, sens_list, spec_list, prec_list, f1_list, auc_list = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for tr_idx, val_idx in skf.split(X_array, y_array):
            X_tr, X_val = X_array[tr_idx], X_array[val_idx]
            y_tr, y_val = y_array[tr_idx], y_array[val_idx]

            pipe = self._build_pipeline(self.config.model_type)
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_val)

            try:
                y_prob = pipe.predict_proba(X_val)[:, 1]
                auc_val = roc_auc_score(y_val, y_prob)
            except Exception:
                auc_val = np.nan

            tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            accs.append(accuracy_score(y_val, y_pred))
            bal_accs.append(balanced_accuracy_score(y_val, y_pred))
            sens_list.append(recall_score(y_val, y_pred, pos_label=1, zero_division=0))
            spec_list.append(spec)
            prec_list.append(precision_score(y_val, y_pred, pos_label=1, zero_division=0))
            f1_list.append(f1_score(y_val, y_pred, pos_label=1, zero_division=0))
            auc_list.append(auc_val)

        return {
            "accuracy": {"mean": float(np.mean(accs)), "std": float(np.std(accs))},
            "balanced_accuracy": {"mean": float(np.mean(bal_accs)), "std": float(np.std(bal_accs))},
            "sensitivity": {"mean": float(np.mean(sens_list)), "std": float(np.std(sens_list))},
            "specificity": {"mean": float(np.mean(spec_list)), "std": float(np.std(spec_list))},
            "precision": {"mean": float(np.mean(prec_list)), "std": float(np.std(prec_list))},
            "f1_score": {"mean": float(np.mean(f1_list)), "std": float(np.std(f1_list))},
            "roc_auc": {"mean": float(np.nanmean(auc_list)), "std": float(np.nanstd(auc_list))},
        }

    # ---------------------------------------------------------
    # 6. HYPERPARAMETER TUNING
    # ---------------------------------------------------------

    def tune_hyperparameters(
        self,
        X: ArrayInput,
        y: np.ndarray,
        param_grid: Optional[Dict] = None,
        scoring: str = "balanced_accuracy",
        n_splits: Optional[int] = None,
    ) -> Dict:
        """
        Hyperparameter tuning via GridSearchCV with scoring='balanced_accuracy'.
        Preprocessing remains inside the pipeline across all CV folds.
        """
        X_array, feature_names = self._validate_X(X)
        y_array = self._validate_y(y, X_array.shape[0])

        self.feature_names_ = feature_names
        n_splits = n_splits or self.config.n_splits

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )

        pipeline = create_gaussian_rbf_svm_pipeline(
            random_state=self.config.random_state,
            class_weight="balanced",
        )

        if param_grid is None:
            param_grid = {
                "svm__C": [0.01, 0.1, 1, 10, 100, 1000],
                "svm__gamma": ["scale", 0.0001, 0.001, 0.01, 0.1, 1],
            }

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )

        start_time = time.time()
        grid.fit(X_array, y_array)
        self.training_time_sec_ = time.time() - start_time

        self.pipeline = grid.best_estimator_
        self.best_params_ = grid.best_params_
        self.is_fitted_ = True

        return {
            "best_params": grid.best_params_,
            "best_score": float(grid.best_score_),
            "cv_results": grid.cv_results_,
        }

    # ---------------------------------------------------------
    # 7. EVALUATION ON TEST DATA
    # ---------------------------------------------------------

    def evaluate(self, X: ArrayInput, y: np.ndarray) -> Dict[str, Union[float, int]]:
        """Evaluate fitted model on test data without modifying weights."""
        X_array, _ = self._validate_X(X)
        y_array = self._validate_y(y, X_array.shape[0])

        start_pred = time.time()
        y_pred = self.predict(X_array)
        pred_time = time.time() - start_pred

        y_prob = self.predict_proba(X_array)[:, 1]

        acc = accuracy_score(y_array, y_pred)
        bal_acc = balanced_accuracy_score(y_array, y_pred)
        sens = recall_score(y_array, y_pred, pos_label=1, zero_division=0)
        prec = precision_score(y_array, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_array, y_pred, pos_label=1, zero_division=0)

        cm = confusion_matrix(y_array, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        try:
            auc_val = float(roc_auc_score(y_array, y_prob))
        except Exception:
            auc_val = float("nan")

        return {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "precision": float(prec),
            "f1_score": float(f1),
            "roc_auc": auc_val,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "prediction_time_sec": float(pred_time),
        }

    # ---------------------------------------------------------
    # 8. SAVE & LOAD
    # ---------------------------------------------------------

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Save pipeline and metadata to disk."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted model.")

        if filepath is None:
            filepath = Path(MODELS_DIR) / "gaussian_rbf_svm.joblib"
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "pipeline": self.pipeline,
            "feature_names": self.feature_names_,
            "canonical_features": CANONICAL_FEATURES,
            "best_params": self.best_params_,
            "config": self.config,
            "training_time_sec": self.training_time_sec_,
            "is_fitted": self.is_fitted_,
        }
        joblib.dump(payload, filepath)
        return filepath

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "EHGClassifier":
        """Load trained model checkpoint (supporting payload dict, Pipeline, or legacy SVC)."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = joblib.load(filepath)
        instance = cls()

        if isinstance(checkpoint, dict) and "pipeline" in checkpoint:
            instance.pipeline = checkpoint["pipeline"]
            instance.feature_names_ = checkpoint.get("feature_names")
            instance.best_params_ = checkpoint.get("best_params")
            instance.config = checkpoint.get("config", instance.config)
            instance.training_time_sec_ = checkpoint.get("training_time_sec")
            instance.is_fitted_ = checkpoint.get("is_fitted", True)
        elif isinstance(checkpoint, Pipeline):
            instance.pipeline = checkpoint
            instance.is_fitted_ = True
        elif isinstance(checkpoint, SVC):
            instance.pipeline = Pipeline(
                steps=[
                    ("scaler", getattr(checkpoint, "scaler_", StandardScaler())),
                    ("svm", checkpoint),
                ]
            )
            instance.is_fitted_ = True
            instance.feature_names_ = getattr(checkpoint, "feature_names_", None)
            instance.best_params_ = getattr(checkpoint, "best_params_", None)
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            scaler_obj = checkpoint.get("scaler", StandardScaler())
            model_obj = checkpoint["model"]
            instance.pipeline = Pipeline(
                steps=[
                    ("scaler", scaler_obj),
                    ("svm", model_obj),
                ]
            )
            instance.is_fitted_ = checkpoint.get("is_fitted", True)
            instance.feature_names_ = checkpoint.get("feature_names")
            instance.best_params_ = checkpoint.get("best_params")
        else:
            raise TypeError(f"Unrecognized checkpoint format: {type(checkpoint)}")

        return instance


# Canonical alias for Gaussian RBF-SVM
GaussianRBFSVMClassifier = EHGClassifier