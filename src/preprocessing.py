"""
src/preprocessing.py
====================
Biomedical signal preprocessing, AAMI EC57 annotation mapping,
and heartbeat segmentation for the MIT-BIH Arrhythmia Database.

AAMI EC57 Standard Classes:
- N: Normal / Non-ectopic (N, L, R, e, j)
- S: Supraventricular ectopic (A, a, J, S)
- V: Ventricular ectopic (V, E)
- F: Fusion of ventricular and normal (F)
- Q: Paced / Unknown / Unclassifiable (/, f, Q, ?)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.signal as signal
import wfdb

from src.data_loader import (
    DS1_TRAIN_RECORDS,
    DS2_TEST_RECORDS,
    PACED_EXCLUDED_RECORDS,
    get_default_data_dir,
    load_record,
)


# Official AAMI EC57 Annotation Symbol to 5-Class Super-Group Mapping
AAMI_MAPPING: Dict[str, str] = {
    # N - Non-ectopic beats
    "N": "N",  # Normal beat
    "L": "N",  # Left bundle branch block beat
    "R": "N",  # Right bundle branch block beat
    "e": "N",  # Atrial escape beat
    "j": "N",  # Nodal (junctional) escape beat

    # S - Supraventricular ectopic beats (SVEB)
    "A": "S",  # Atrial premature beat
    "a": "S",  # Aberrated atrial premature beat
    "J": "S",  # Nodal (junctional) premature beat
    "S": "S",  # Supraventricular premature or ectopic beat

    # V - Ventricular ectopic beats (VEB)
    "V": "V",  # Premature ventricular contraction
    "E": "V",  # Ventricular escape beat

    # F - Fusion beats
    "F": "F",  # Fusion of ventricular and normal beat

    # Q - Paced / Unclassifiable beats
    "/": "Q",  # Paced beat
    "f": "Q",  # Fusion of paced and normal beat
    "Q": "Q",  # Unclassifiable beat
    "?": "Q",  # Beat not classified during learning
}

# Non-beat annotations defined by PhysioNet (must be excluded from beat classification)
NON_BEAT_SYMBOLS = {
    "+", "~", "|", "[", "]", "x", "!", '"', "@", "p", "t", "u",
    "`", "'", "^", "s", "T", "*"
}

AAMI_CLASSES: List[str] = ["N", "S", "V", "F", "Q"]
CLASS_TO_IDX: Dict[str, int] = {cls: idx for idx, cls in enumerate(AAMI_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: cls for idx, cls in enumerate(AAMI_CLASSES)}


def bandpass_filter(
    ecg_signal: np.ndarray,
    fs: float = 360.0,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 3
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter to remove baseline wander
    (< 0.5 Hz) and powerline/high-frequency muscle noise (> 40.0 Hz).

    Zero-phase filtering (filtfilt) ensures no phase distortion or temporal
    displacement of R-peaks is introduced.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="bandpass")
    filtered = signal.filtfilt(b, a, ecg_signal)
    return filtered


def normalize_signal(ecg_signal: np.ndarray, method: str = "z_score") -> np.ndarray:
    """
    Normalize an ECG lead recording.

    Parameters
    ----------
    ecg_signal : np.ndarray
        1D ECG signal array.
    method : str
        'z_score' (mean 0, std 1) or 'min_max' (scaled to [0, 1]).
    """
    if method == "z_score":
        mean = np.nanmean(ecg_signal)
        std = np.nanstd(ecg_signal)
        if std == 0:
            return ecg_signal - mean
        return (ecg_signal - mean) / std

    elif method == "min_max":
        min_val = np.nanmin(ecg_signal)
        max_val = np.nanmax(ecg_signal)
        if max_val == min_val:
            return np.zeros_like(ecg_signal)
        return (ecg_signal - min_val) / (max_val - min_val)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def select_lead(
    record: wfdb.Record,
    preferred_lead: str = "MLII"
) -> Tuple[np.ndarray, str, int]:
    """
    Select the preferred lead (default MLII, present in almost all recordings).
    Falls back to channel 0 if the preferred lead is not found.
    """
    sig_names = record.sig_name
    if preferred_lead in sig_names:
        idx = sig_names.index(preferred_lead)
        return record.p_signal[:, idx], preferred_lead, idx
    else:
        # Fallback to first available channel
        fallback_lead = sig_names[0]
        return record.p_signal[:, 0], fallback_lead, 0


def segment_heartbeats(
    ecg_signal: np.ndarray,
    annotation: wfdb.Annotation,
    pre_r: int = 90,
    post_r: int = 110,
    exclude_unknown_q: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Segment individual heartbeats centered around R-peak annotations.

    Parameters
    ----------
    ecg_signal : np.ndarray
        Preprocessed 1D ECG signal.
    annotation : wfdb.Annotation
        PhysioNet annotation object containing sample indices and symbols.
    pre_r : int
        Number of samples before the R-peak (90 samples = 250 ms at 360 Hz).
    post_r : int
        Number of samples after the R-peak (110 samples = 305 ms at 360 Hz).
    exclude_unknown_q : bool
        If True, excludes 'Q' class beats (often unclassifiable or paced).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]
        - segments: array of shape (N_beats, pre_r + post_r)
        - labels: numeric class index array of shape (N_beats,)
        - r_peak_indices: sample positions in the original recording
        - raw_symbols: list of original PhysioNet annotation symbols
    """
    total_len = len(ecg_signal)
    segments = []
    labels = []
    r_peaks = []
    raw_symbols = []

    for r_idx, sym in zip(annotation.sample, annotation.symbol):
        # Exclude non-beat annotations (rhythm changes, noise, comments)
        if sym in NON_BEAT_SYMBOLS or sym not in AAMI_MAPPING:
            continue

        aami_class = AAMI_MAPPING[sym]
        if exclude_unknown_q and aami_class == "Q":
            continue

        # Check boundary constraints
        start_idx = r_idx - pre_r
        end_idx = r_idx + post_r

        if start_idx < 0 or end_idx > total_len:
            # Skip heartbeats truncated at signal edges
            continue

        beat_window = ecg_signal[start_idx:end_idx]
        segments.append(beat_window)
        labels.append(CLASS_TO_IDX[aami_class])
        r_peaks.append(r_idx)
        raw_symbols.append(sym)

    return (
        np.array(segments, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(r_peaks, dtype=np.int64),
        raw_symbols
    )


def preprocess_record(
    record_name: str,
    data_dir: Optional[Path] = None,
    preferred_lead: str = "MLII",
    lowcut: float = 0.5,
    highcut: float = 40.0,
    pre_r: int = 90,
    post_r: int = 110,
    normalization: str = "z_score"
) -> Dict[str, Any]:
    """
    End-to-end preprocessing pipeline for a single record.

    Loads record -> selects lead -> bandpass filters -> normalizes -> segments heartbeats.
    """
    record, annotation = load_record(record_name, data_dir)
    raw_lead, lead_name, lead_idx = select_lead(record, preferred_lead)

    # 1. Bandpass filter
    filtered_lead = bandpass_filter(
        raw_lead, fs=record.fs, lowcut=lowcut, highcut=highcut
    )

    # 2. Normalization
    normalized_lead = normalize_signal(filtered_lead, method=normalization)

    # 3. Heartbeat segmentation
    segments, labels, r_peaks, raw_syms = segment_heartbeats(
        normalized_lead, annotation, pre_r=pre_r, post_r=post_r
    )

    return {
        "record_name": record_name,
        "lead_name": lead_name,
        "raw_signal": raw_lead,
        "filtered_signal": filtered_lead,
        "normalized_signal": normalized_lead,
        "segments": segments,
        "labels": labels,
        "r_peak_indices": r_peaks,
        "raw_symbols": raw_syms,
        "num_beats": len(labels)
    }


def main():
    parser = argparse.ArgumentParser(
        description="ECG Preprocessing and Heartbeat Segmentation"
    )
    parser.add_argument(
        "--record",
        type=str,
        default="100",
        help="Record ID to process (default: 100)."
    )
    parser.add_argument(
        "--lead",
        type=str,
        default="MLII",
        help="Preferred lead name (default: MLII)."
    )
    args = parser.parse_args()

    data_dir = get_default_data_dir()
    print(f"[Preprocessing] Processing record {args.record}...")

    try:
        results = preprocess_record(args.record, data_dir=data_dir, preferred_lead=args.lead)
        print(f"  Lead used: {results['lead_name']}")
        print(f"  Total heartbeats segmented: {results['num_beats']}")
        print(f"  Segment shape: {results['segments'].shape}")

        unique_labels, counts = np.unique(results["labels"], return_counts=True)
        print("  AAMI Class Distribution for this record:")
        for label_idx, count in zip(unique_labels, counts):
            class_name = IDX_TO_CLASS[label_idx]
            print(f"    - Class {class_name} ({label_idx}): {count} beats ({count / results['num_beats'] * 100:.1f}%)")

    except FileNotFoundError:
        print(f"[Error] Record {args.record} not found in {data_dir}. Run python -m src.data_loader --download-sample first.")


if __name__ == "__main__":
    main()
