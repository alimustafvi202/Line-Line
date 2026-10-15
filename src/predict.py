"""
src/predict.py
==============
Inference and interactive testing script for trained 1D-CNN ECG models.

Allows you to:
1. Test model predictions on any downloaded MIT-BIH record.
2. Compare AI predictions against clinical PhysioNet ground truth.
3. Generate a diagnostic visual plot of predicted heartbeats with confidence scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data_loader import get_default_data_dir
from src.evaluation import get_outputs_dir
from src.model import ECGCNN1D
from src.preprocessing import AAMI_CLASSES, IDX_TO_CLASS, preprocess_record


def load_trained_model(
    model_name: str = "improved_1d_cnn",
    device: Optional[torch.device] = None
) -> ECGCNN1D:
    """Load trained 1D-CNN weights from outputs/models/."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(__file__).resolve().parent.parent / "outputs" / "models"
    model_path = model_dir / f"{model_name}.pt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at: {model_path}. "
            f"Available models in {model_dir}: {[f.name for f in model_dir.glob('*.pt')]}"
        )

    model = ECGCNN1D(in_channels=1, num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location=dev))
    model.to(dev)
    model.eval()
    return model


def predict_record(
    record_name: str = "100",
    model_name: str = "improved_1d_cnn"
):
    """
    Run inference on an entire ECG recording, print prediction stats,
    and save a diagnostic visualization plot.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = get_default_data_dir()
    _, fig_dir = get_outputs_dir()

    print("=" * 65)
    print(f"AI ECG Testing & Inference: Record {record_name}")
    print(f"Model: {model_name} (Device: {device})")
    print("=" * 65)

    # 1. Preprocess recording
    print(f"\n[Step 1] Loading and preprocessing Record {record_name}...")
    rec_data = preprocess_record(record_name, data_dir=data_dir)
    segments = rec_data["segments"]  # (N, 200)
    y_true = rec_data["labels"]      # (N,)
    r_peaks = rec_data["r_peak_indices"]

    print(f"  - Total heartbeats extracted: {len(segments):,}")
    print(f"  - Lead used: {rec_data['lead_name']}")

    # 2. Load model
    print(f"\n[Step 2] Loading model weights ({model_name}.pt)...")
    model = load_trained_model(model_name, device=device)

    # 3. Model Inference
    print(f"\n[Step 3] Running neural network forward pass...")
    tensor_x = torch.tensor(segments, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        logits = model(tensor_x)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()
        y_pred = np.argmax(probabilities, axis=1)

    # 4. Comparative Statistics
    print("\n--- Diagnostic Prediction Summary ---")
    correct = np.sum(y_pred == y_true)
    total = len(y_true)
    accuracy = correct / total * 100

    print(f"Overall Record Accuracy: {accuracy:.2f}% ({correct:,}/{total:,} beats correct)")
    print("\nClass Breakdown Comparison:")
    print(f"{'Class':<10} | {'Ground Truth':<15} | {'AI Predicted':<15} | {'Detection Status'}")
    print("-" * 65)

    for idx, cls in enumerate(AAMI_CLASSES):
        true_cnt = int(np.sum(y_true == idx))
        pred_cnt = int(np.sum(y_pred == idx))
        status = "Match" if true_cnt == pred_cnt else "Variance"
        if true_cnt > 0 and pred_cnt > 0:
            status = "Detected"
        elif true_cnt > 0 and pred_cnt == 0:
            status = "Missed"
        elif true_cnt == 0 and pred_cnt > 0:
            status = "False Positives"
        print(f"Class {cls:<4} | {true_cnt:<15} | {pred_cnt:<15} | {status}")

    # 5. Visual Inspection of Sample Beats
    print(f"\n[Step 4] Generating visual prediction plot for sample heartbeats...")
    fig_path = fig_dir / f"test_prediction_record_{record_name}.png"

    # Select interesting beats (e.g. some Normal and any ectopic beats)
    ectopic_indices = np.where(y_true != 0)[0]
    normal_indices = np.where(y_true == 0)[0]

    sample_indices = []
    if len(ectopic_indices) > 0:
        sample_indices.extend(ectopic_indices[:3])
    if len(normal_indices) > 0:
        sample_indices.extend(normal_indices[:3])

    sample_indices = sample_indices[:6]
    fs = 360.0
    time_ms = (np.arange(200) - 90) / fs * 1000.0

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    fig.suptitle(
        f"Sample AI Heartbeat Predictions on Record {record_name} ({model_name})",
        fontsize=14,
        fontweight="bold"
    )

    for i, beat_idx in enumerate(sample_indices):
        ax = axes[i]
        waveform = segments[beat_idx]
        true_cls = IDX_TO_CLASS[y_true[beat_idx]]
        pred_cls = IDX_TO_CLASS[y_pred[beat_idx]]
        conf = probabilities[beat_idx, y_pred[beat_idx]] * 100

        is_correct = (true_cls == pred_cls)
        color = "#1f77b4" if is_correct else "#d62728"

        ax.plot(time_ms, waveform, color=color, linewidth=1.8)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.6)
        ax.set_title(
            f"Beat #{beat_idx:,}\nTrue: {true_cls} | AI: {pred_cls} ({conf:.1f}% conf)",
            fontsize=10,
            fontweight="bold",
            color="green" if is_correct else "red"
        )
        ax.set_xlabel("Time from R-peak (ms)", fontsize=9)
        ax.set_ylabel("Norm Amplitude", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"[Visualization] Prediction diagnostic plot saved to:\n  -> {fig_path}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Test trained ECG models on patients")
    parser.add_argument("--record", type=str, default="100", help="Record ID to test (e.g. 100, 101, 106, 119)")
    parser.add_argument(
        "--model",
        type=str,
        default="improved_1d_cnn",
        choices=["baseline_1d_cnn", "improved_1d_cnn", "multi_arrhythmia_1d_cnn"],
        help="Model checkpoint to test"
    )
    args = parser.parse_args()
    predict_record(args.record, args.model)


if __name__ == "__main__":
    main()
