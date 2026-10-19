"""
Evaluation module for model performance metrics and visualizations.

Computes medical metrics (Sensitivity/Recall, Specificity, ROC-AUC)
and outputs plots (Confusion Matrix, ROC Curve, IMF Decomposition).
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve, precision_recall_curve, auc
)

from src.config import PLOTS_DIR


class EHGEvaluator:
    """Evaluation suite for clinical EHG triage models."""

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute full clinical triage metrics suite.
        
        Sensitivity = Recall = TP / (TP + FN) (ability to detect Preterm)
        Specificity = TN / (TN + FP) (ability to identify Term)
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        roc_auc = 0.5
        if y_proba is not None:
            # Handle binary vs probability array shape
            pos_proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            roc_auc = float(roc_auc_score(y_true, pos_proba))

        return {
            'accuracy': float(acc),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Triage Confusion Matrix",
        save_path: Optional[Union[str, Path]] = None
    ):
        """Plot and save confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Term (0)', 'Preterm (1)'],
            yticklabels=['Term (0)', 'Preterm (1)']
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(title)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "Preterm Triage ROC Curve",
        save_path: Optional[Union[str, Path]] = None
    ):
        """Plot and save Receiver Operating Characteristic (ROC) curve."""
        pos_proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        fpr, tpr, _ = roc_curve(y_true, pos_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_imfs(
        signal: np.ndarray,
        imfs: np.ndarray,
        fs: float = 20.0,
        title: str = "EMD Intrinsic Mode Functions",
        save_path: Optional[Union[str, Path]] = None
    ):
        """Plot raw signal and decomposed IMFs stack."""
        n_imfs = len(imfs)
        time = np.arange(len(signal)) / fs

        fig, axes = plt.subplots(n_imfs + 1, 1, figsize=(10, 2 * (n_imfs + 1)), sharex=True)
        axes[0].plot(time, signal, color='black', lw=1)
        axes[0].set_ylabel('Raw EHG')
        axes[0].set_title(title)

        for i in range(n_imfs):
            axes[i + 1].plot(time, imfs[i], color=f'C{i%10}', lw=1)
            axes[i + 1].set_ylabel(f'IMF {i+1}')

        axes[-1].set_xlabel('Time (seconds)')
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
