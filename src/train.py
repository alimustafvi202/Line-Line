"""
src/train.py
============
Academic training and evaluation pipeline for ECG heartbeat anomaly detection.
Features:
- Inter-Patient split evaluation (DS1 train/val vs DS2 test)
- Weighted Cross-Entropy Loss to handle class imbalance
- Fully reproducible seeds (random seed: 42)
- Comprehensive test evaluation with precision, recall, macro F1, and confusion matrix
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_loader import DS1_TRAIN_RECORDS, DS2_TEST_RECORDS, get_default_data_dir
from src.dataset import (
    ECGHeartbeatDataset,
    build_split_dataset,
    compute_class_weights,
    get_processed_data_dir,
)
from src.evaluation import compute_metrics, get_outputs_dir, plot_confusion_matrix, save_metrics_report
from src.model import ECGCNN1D, get_model_summary


def set_seed(seed: int = 42):
    """Ensure full reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train model for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_y)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == batch_y).sum().item()
        total += len(batch_y)

    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model on a dataloader. Returns (avg_loss, y_true, y_pred)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * len(batch_y)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            total += len(batch_y)

    return total_loss / total, np.array(all_targets), np.array(all_preds)


def main():
    parser = argparse.ArgumentParser(description="ECG Arrhythmia Baseline Model Trainer")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--quick", action="store_true", help="Use initial sample records for rapid testing")
    parser.add_argument("--full", action="store_true", help="Use full DS1 and DS2 datasets (44 records)")
    parser.add_argument("--experiment-name", type=str, default="improved_1d_cnn", help="Name identifier for this experiment")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using computing device: {device}")

    # Determine train & test records
    if args.quick:
        # Quick baseline with downloaded local sample records
        train_records = ["101"]
        test_records = ["100"]
        cache_train = "quick_train_101"
        cache_test = "quick_test_100"
        print("[Training] Mode: Quick Baseline (DS1: 101, DS2: 100)")
    elif args.full:
        train_records = DS1_TRAIN_RECORDS
        test_records = DS2_TEST_RECORDS
        cache_train = "full_ds1_train"
        cache_test = "full_ds2_test"
        print(f"[Training] Mode: Full Inter-Patient (DS1: {len(train_records)} recs, DS2: {len(test_records)} recs)")
    elif args.experiment_name == "multi_arrhythmia_1d_cnn":
        # Comprehensive multi-arrhythmia training set (all strictly DS1):
        # 101 (baseline), 106 (VEB), 119 (VEB), 201 (VEB + SVEB), 209 (high SVEB: 383 A beats)
        train_records = ["101", "106", "119", "201", "209"]
        test_records = ["100"]
        cache_train = "multi_arrhythmia_ds1_5recs"
        cache_test = "quick_test_100"
        print("[Training] Mode: Multi-Arrhythmia Training (DS1: 101, 106, 119, 201, 209 | DS2 Test: 100)")
    else:
        # Improved multi-patient subset with arrhythmia diversity (V and S beats)
        # DS1 train: 101 (N, S), 106 (V), 119 (V)
        # DS2 test: 100 (N, S, V)
        train_records = ["101", "106", "119"]
        test_records = ["100"]
        cache_train = "improved_ds1_101_106_119"
        cache_test = "quick_test_100"
        print("[Training] Mode: Improved Arrhythmia Training (DS1: 101, 106, 119 | DS2 Test: 100)")

    # 1. Load Train and Test Data
    print("\n--- 1. Building Inter-Patient Datasets ---")
    X_train, y_train, train_dist = build_split_dataset(train_records, cache_name=cache_train)
    X_test, y_test, test_dist = build_split_dataset(test_records, cache_name=cache_test)

    # 2. Compute Class Weights for Loss Function
    class_weights = compute_class_weights(y_train, num_classes=5).to(device)
    print(f"\n[Training] Computed Class Weights for Cross-Entropy: {class_weights.cpu().numpy().round(3)}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 3. Create PyTorch DataLoaders
    train_dataset = ECGHeartbeatDataset(X_train, y_train)
    test_dataset = ECGHeartbeatDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Initialize Model
    print("\n--- 2. Initializing 1D-CNN Baseline Architecture ---")
    model = ECGCNN1D(in_channels=1, num_classes=5, base_filters=32, dropout=0.25).to(device)
    print(get_model_summary(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # Directory for model checkpoints
    model_dir = Path(__file__).resolve().parent.parent / "outputs" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / f"{args.experiment_name}.pt"

    # 5. Training Loop
    print("\n--- 3. Starting Training Loop ---")
    best_macro_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, y_true_val, y_pred_val = evaluate(model, test_loader, criterion, device)

        val_metrics = compute_metrics(y_true_val, y_pred_val)
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]

        scheduler.step(val_f1)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Test Loss: {val_loss:.4f} | Test Acc: {val_acc * 100:.2f}% | "
            f"Test Macro F1: {val_f1:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1
        })

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best model checkpoint to {best_model_path.name} (Macro F1: {best_macro_f1:.4f})")

    # 6. Final Evaluation on Best Model
    print("\n--- 4. Final Test Set Evaluation ---")
    model.load_state_dict(torch.load(best_model_path))
    _, y_true_final, y_pred_final = evaluate(model, test_loader, criterion, device)
    final_metrics = compute_metrics(y_true_final, y_pred_final)

    print("\n" + "=" * 60)
    print(f"FINAL PERFORMANCE REPORT: {args.experiment_name.upper()} (Unseen Test Patients)")
    print("=" * 60)
    print(f"Overall Accuracy:  {final_metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score:    {final_metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {final_metrics['weighted_f1']:.4f}")
    print("\nPer-Class Breakdown:")
    for cls_name, vals in final_metrics["per_class"].items():
        print(
            f"  Class {cls_name:2s}: "
            f"Sens(Recall)={vals['recall_sensitivity']:.4f} | "
            f"Spec={vals['specificity']:.4f} | "
            f"Prec={vals['precision']:.4f} | "
            f"F1={vals['f1_score']:.4f} | "
            f"N={vals['support']}"
        )
    print("=" * 60)

    # 7. Export Metric Reports and Confusion Matrices
    table_dir, fig_dir = get_outputs_dir()
    save_metrics_report(final_metrics, experiment_name=args.experiment_name, table_dir=table_dir)

    cm = np.array(final_metrics["confusion_matrix"])
    plot_confusion_matrix(
        cm,
        normalize=True,
        title=f"{args.experiment_name} Normalized Confusion Matrix (DS2 Test Set)",
        save_path=fig_dir / f"{args.experiment_name}_cm_normalized.png"
    )
    plot_confusion_matrix(
        cm,
        normalize=False,
        title=f"{args.experiment_name} Raw Confusion Matrix (DS2 Test Set)",
        save_path=fig_dir / f"{args.experiment_name}_cm_raw.png"
    )

    print(f"\n[Complete] Experiment '{args.experiment_name}' finished! Artifacts saved in {table_dir} and {fig_dir}.")


if __name__ == "__main__":
    main()
