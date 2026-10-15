"""
src/dataset.py
==============
PyTorch Dataset and patient-wise data partitioning for
the MIT-BIH Arrhythmia Database.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_loader import (
    ALL_MITDB_RECORDS,
    DS1_TRAIN_RECORDS,
    DS2_TEST_RECORDS,
    download_records,
    get_default_data_dir,
)
from src.preprocessing import AAMI_CLASSES, preprocess_record


def get_processed_data_dir() -> Path:
    """Return the absolute path to data/processed/."""
    base_dir = Path(__file__).resolve().parent.parent
    proc_dir = base_dir / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    return proc_dir


class ECGHeartbeatDataset(Dataset):
    """
    PyTorch Dataset wrapper for segmented ECG heartbeats.

    Parameters
    ----------
    segments : np.ndarray or torch.Tensor
        Array of shape (N, 200) or (N, 1, 200).
    labels : np.ndarray or torch.Tensor
        Array of integer class labels of shape (N,).
    """
    def __init__(self, segments: np.ndarray, labels: np.ndarray):
        if segments.ndim == 2:
            segments = np.expand_dims(segments, axis=1)  # (N, 200) -> (N, 1, 200)

        self.segments = torch.tensor(segments, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.segments[idx], self.labels[idx]


def build_split_dataset(
    record_names: List[str],
    data_dir: Optional[Path] = None,
    cache_name: Optional[str] = None,
    preferred_lead: str = "MLII"
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Load, preprocess, and aggregate heartbeats from specified patient records.

    Parameters
    ----------
    record_names : List[str]
        List of MIT-BIH record IDs to include.
    data_dir : Optional[Path]
        Directory where raw records are located.
    cache_name : Optional[str]
        If given, caches/loads the processed data to/from data/processed/{cache_name}.npz.
    preferred_lead : str
        ECG lead name (default: MLII).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict[str, int]]
        - X: concatenated segments array (N, 200)
        - y: concatenated integer class labels (N,)
        - class_counts: dictionary of beat counts per AAMI class
    """
    proc_dir = get_processed_data_dir()
    cache_file = proc_dir / f"{cache_name}.npz" if cache_name else None

    # Load from pre-processed cache if available
    if cache_file and cache_file.exists():
        print(f"[Dataset] Loading cached split from {cache_file}...")
        data = np.load(str(cache_file))
        X = data["segments"]
        y = data["labels"]
        counts = {cls: int(np.sum(y == idx)) for idx, cls in enumerate(AAMI_CLASSES)}
        return X, y, counts

    raw_dir = data_dir or get_default_data_dir()
    all_segments: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    print(f"[Dataset] Building dataset from {len(record_names)} records: {record_names}")

    # Ensure all required records are downloaded
    needed_downloads = [
        rec for rec in record_names
        if not (raw_dir / f"{rec}.dat").exists() or not (raw_dir / f"{rec}.hea").exists()
    ]
    if needed_downloads:
        print(f"[Dataset] Missing {len(needed_downloads)} record(s). Downloading...")
        download_records(needed_downloads, output_dir=raw_dir)

    for rec_id in record_names:
        try:
            res = preprocess_record(rec_id, data_dir=raw_dir, preferred_lead=preferred_lead)
            if res["num_beats"] > 0:
                all_segments.append(res["segments"])
                all_labels.append(res["labels"])
        except Exception as exc:
            print(f"[Warning] Failed to preprocess record {rec_id}: {exc}")

    if not all_segments:
        raise ValueError("No heartbeats could be extracted from the specified records.")

    X = np.concatenate(all_segments, axis=0)
    y = np.concatenate(all_labels, axis=0)

    counts = {cls: int(np.sum(y == idx)) for idx, cls in enumerate(AAMI_CLASSES)}
    print(f"[Dataset] Extracted {len(y):,} total heartbeats. Distribution: {counts}")

    if cache_file:
        print(f"[Dataset] Saving cache to {cache_file}...")
        np.savez_compressed(str(cache_file), segments=X, labels=y)

    return X, y, counts


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int = 5,
    beta: float = 0.999
) -> torch.Tensor:
    """
    Compute class weights to balance Cross-Entropy Loss for imbalanced classes.
    Uses inverse frequency with smoothing.
    """
    counts = np.bincount(labels, minlength=num_classes)
    weights = np.zeros(num_classes, dtype=np.float32)

    for i in range(num_classes):
        if counts[i] > 0:
            weights[i] = len(labels) / (num_classes * counts[i])
        else:
            weights[i] = 1.0

    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)
