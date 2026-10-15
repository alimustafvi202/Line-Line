"""
src/evaluation.py
=================
Comprehensive classification evaluation module for academic biomedical research.

Computes:
- Overall Accuracy
- Per-class Precision, Recall (Sensitivity), Specificity, and F1-score
- Macro-averaged and Weighted-averaged F1-scores
- Raw and Normalized Confusion Matrices
- Exports structured reports to outputs/tables/ and figures to outputs/figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.preprocessing import AAMI_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS


def get_outputs_dir() -> Tuple[Path, Path]:
    """Return paths to outputs/tables/ and outputs/figures/."""
    base_dir = Path(__file__).resolve().parent.parent
    table_dir = base_dir / "outputs" / "tables"
    fig_dir = base_dir / "outputs" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, fig_dir


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth integer class labels.
    y_pred : np.ndarray
        Predicted integer class labels.
    class_names : Optional[List[str]]
        Names corresponding to each class index.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing overall, per-class, and averaged metrics.
    """
    classes = class_names or AAMI_CLASSES
    n_classes = len(classes)
    acc = float(accuracy_score(y_true, y_pred))

    # Precision, recall, f1 per class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    # Specificity per class (TN / (TN + FP))
    specificity = []
    for i in range(n_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        specificity.append(spec)

    per_class_metrics = {}
    for i, cls_name in enumerate(classes):
        per_class_metrics[cls_name] = {
            "precision": float(precision[i]),
            "recall_sensitivity": float(recall[i]),
            "specificity": float(specificity[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i])
        }

    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.average(f1, weights=support)) if np.sum(support) > 0 else 0.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "per_class": per_class_metrics
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "AAMI EC57 Confusion Matrix",
    save_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot and save the confusion matrix heatmap.
    """
    classes = class_names or AAMI_CLASSES
    fig, ax = plt.subplots(figsize=(7, 6))

    if normalize:
        # Avoid division by zero for classes with 0 support
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        display_cm = cm.astype("float") / row_sums
        fmt = ".2%"
        cbar_label = "Normalized Proportion"
    else:
        display_cm = cm
        fmt = "d"
        cbar_label = "Sample Count"

    cax = ax.imshow(display_cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(cax, ax=ax, label=cbar_label)

    ax.set_title(title, fontsize=13, fontweight="bold")
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10)

    # Annotate numbers in cells
    thresh = display_cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_str = f"{display_cm[i, j]:{fmt}}" if normalize else f"{cm[i, j]:,}"
            ax.text(
                j, i, val_str,
                ha="center", va="center",
                color="white" if display_cm[i, j] > thresh else "black",
                fontsize=10,
                fontweight="bold"
            )

    ax.set_ylabel("True Ground Truth Label", fontsize=11)
    ax.set_xlabel("Predicted Model Label", fontsize=11)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Evaluation] Confusion matrix saved to {save_path}")

    if show:
        plt.show()

    return fig


def save_metrics_report(
    metrics: Dict[str, Any],
    experiment_name: str = "baseline",
    table_dir: Optional[Path] = None
) -> Tuple[Path, Path]:
    """
    Export metrics dictionary into JSON and tabular CSV summaries.
    """
    t_dir, _ = get_outputs_dir()
    target_dir = table_dir or t_dir

    json_path = target_dir / f"{experiment_name}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Create CSV tabular summary
    rows = []
    for cls_name, vals in metrics["per_class"].items():
        rows.append({
            "Class": cls_name,
            "Precision": f"{vals['precision']:.4f}",
            "Recall (Sensitivity)": f"{vals['recall_sensitivity']:.4f}",
            "Specificity": f"{vals['specificity']:.4f}",
            "F1-Score": f"{vals['f1_score']:.4f}",
            "Support (Beats)": vals["support"]
        })

    # Summary row
    rows.append({
        "Class": "OVERALL (Macro)",
        "Precision": "-",
        "Recall (Sensitivity)": "-",
        "Specificity": "-",
        "F1-Score": f"{metrics['macro_f1']:.4f}",
        "Support (Beats)": sum(vals["support"] for vals in metrics["per_class"].values())
    })
    rows.append({
        "Class": "OVERALL (Accuracy)",
        "Precision": "-",
        "Recall (Sensitivity)": "-",
        "Specificity": "-",
        "F1-Score": f"{metrics['accuracy']:.4f}",
        "Support (Beats)": sum(vals["support"] for vals in metrics["per_class"].values())
    })

    df = pd.DataFrame(rows)
    csv_path = target_dir / f"{experiment_name}_metrics.csv"
    df.to_csv(csv_path, index=False)

    print(f"[Evaluation] Metrics report saved to:\n  - JSON: {json_path}\n  - CSV:  {csv_path}")
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="ECG Evaluation & Metrics Module")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run verification test with synthetic validation distributions to test evaluation pipeline."
    )
    args = parser.parse_args()

    table_dir, fig_dir = get_outputs_dir()

    if args.demo:
        print("[Evaluation] Running evaluation test pipeline...")
        np.random.seed(42)

        # Realistic mock class imbalance (mostly N, fewer S, V, F, Q)
        y_true = np.random.choice([0, 1, 2, 3, 4], size=1000, p=[0.85, 0.05, 0.07, 0.02, 0.01])
        # Add 10% mock classification errors
        y_pred = y_true.copy()
        flip_indices = np.random.choice(1000, size=100, replace=False)
        y_pred[flip_indices] = np.random.choice([0, 1, 2, 3, 4], size=100)

        metrics = compute_metrics(y_true, y_pred)
        print("\n--- Summary Performance Metrics ---")
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1-Score:   {metrics['macro_f1']:.4f}")
        print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")

        # Save reports
        save_metrics_report(metrics, experiment_name="pipeline_verification_demo", table_dir=table_dir)

        # Plot confusion matrices
        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(
            cm,
            normalize=True,
            title="Verification Demo: Normalized Confusion Matrix",
            save_path=fig_dir / "pipeline_verification_cm_normalized.png"
        )
        plot_confusion_matrix(
            cm,
            normalize=False,
            title="Verification Demo: Raw Counts Confusion Matrix",
            save_path=fig_dir / "pipeline_verification_cm_raw.png"
        )
        print("[Evaluation] Verification test successfully completed.")


if __name__ == "__main__":
    main()
