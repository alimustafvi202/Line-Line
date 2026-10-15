"""
src/visualization.py
====================
Academic biomedical visualization module for ECG signal inspection,
filter comparisons, heartbeat morphologies, and class distribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import get_default_data_dir
from src.preprocessing import (
    AAMI_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    preprocess_record,
)


def get_figures_dir() -> Path:
    """Return the absolute path to outputs/figures/."""
    base_dir = Path(__file__).resolve().parent.parent
    fig_dir = base_dir / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def plot_signal_filtering_comparison(
    record_dict: Dict,
    start_sec: float = 0.0,
    duration_sec: float = 6.0,
    save_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot raw vs. bandpass filtered & normalized ECG waveforms with R-peaks.

    Parameters
    ----------
    record_dict : Dict
        Output dictionary from preprocess_record().
    start_sec : float
        Start timestamp in seconds.
    duration_sec : float
        Duration of the window to display in seconds.
    save_path : Optional[Path]
        If provided, saves figure to this path.
    show : bool
        Whether to call plt.show().
    """
    fs = 360.0
    start_sample = int(start_sec * fs)
    end_sample = int((start_sec + duration_sec) * fs)
    time_axis = np.arange(start_sample, end_sample) / fs

    raw_segment = record_dict["raw_signal"][start_sample:end_sample]
    filt_segment = record_dict["filtered_signal"][start_sample:end_sample]
    norm_segment = record_dict["normalized_signal"][start_sample:end_sample]

    # Find R-peaks within this window
    r_peaks = record_dict["r_peak_indices"]
    labels = record_dict["labels"]
    mask = (r_peaks >= start_sample) & (r_peaks < end_sample)
    window_peaks = r_peaks[mask]
    window_labels = labels[mask]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        f"ECG Signal Inspection & Preprocessing - Record {record_dict['record_name']} ({record_dict['lead_name']})",
        fontsize=14,
        fontweight="bold"
    )

    # 1. Raw ECG
    axes[0].plot(time_axis, raw_segment, color="#1f77b4", linewidth=1.2, label="Raw ECG")
    axes[0].set_ylabel("Amplitude (mV)", fontsize=11)
    axes[0].set_title("Raw ECG Signal (Includes baseline wander & high-frequency noise)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right")

    # 2. Filtered ECG
    axes[1].plot(time_axis, filt_segment, color="#2ca02c", linewidth=1.2, label="Bandpass (0.5 - 40 Hz)")
    axes[1].set_ylabel("Amplitude (mV)", fontsize=11)
    axes[1].set_title("Zero-Phase Bandpass Filtered (Baseline wander & EMG noise eliminated)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right")

    # 3. Normalized ECG with R-Peak markers
    axes[2].plot(time_axis, norm_segment, color="#9467bd", linewidth=1.2, label="Z-score Normalized")
    peak_times = window_peaks / fs
    peak_amplitudes = record_dict["normalized_signal"][window_peaks]
    axes[2].scatter(peak_times, peak_amplitudes, color="#d62728", s=60, zorder=5, label="Annotated R-peak")

    for pt, pa, lbl in zip(peak_times, peak_amplitudes, window_labels):
        cls_name = IDX_TO_CLASS[lbl]
        axes[2].annotate(
            f"{cls_name}",
            (pt, pa),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
            color="#d62728"
        )

    axes[2].set_xlabel("Time (seconds)", fontsize=11)
    axes[2].set_ylabel("Normalized Amplitude (Z)", fontsize=11)
    axes[2].set_title("Normalized Signal with AAMI Annotations", fontsize=11)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Visualization] Figure saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_heartbeat_morphologies(
    segments: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot the mean waveform and +/- 1 standard deviation for each present AAMI class.
    """
    fs = 360.0
    pre_r = 90
    time_ms = (np.arange(segments.shape[1]) - pre_r) / fs * 1000.0  # centered at R-peak in ms

    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)

    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4), sharey=True)
    if n_classes == 1:
        axes = [axes]

    fig.suptitle(
        "Mean Heartbeat Morphologies per AAMI Class (± 1 Standard Deviation)",
        fontsize=14,
        fontweight="bold"
    )

    palette = {
        "N": "#1f77b4",
        "S": "#ff7f0e",
        "V": "#d62728",
        "F": "#2ca02c",
        "Q": "#9467bd"
    }

    for ax, cls_idx in zip(axes, unique_classes):
        cls_name = IDX_TO_CLASS[cls_idx]
        cls_segments = segments[labels == cls_idx]
        mean_wave = np.mean(cls_segments, axis=0)
        std_wave = np.std(cls_segments, axis=0)
        color = palette.get(cls_name, "#333333")

        # Plot individual faint traces (up to 20)
        for i in range(min(15, len(cls_segments))):
            ax.plot(time_ms, cls_segments[i], color=color, alpha=0.15, linewidth=0.8)

        # Plot mean and std band
        ax.plot(time_ms, mean_wave, color=color, linewidth=2.0, label=f"Mean {cls_name}")
        ax.fill_between(time_ms, mean_wave - std_wave, mean_wave + std_wave, color=color, alpha=0.25)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.7)
        ax.set_title(f"Class {cls_name} (N={len(cls_segments)})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time from R-peak (ms)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")

    axes[0].set_ylabel("Normalized Amplitude (Z)", fontsize=11)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Visualization] Figure saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_class_distribution(
    labels: np.ndarray,
    title_suffix: str = "",
    save_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot bar charts of class distribution with absolute counts and percentages.
    """
    total = len(labels)
    counts = [int(np.sum(labels == CLASS_TO_IDX[c])) for c in AAMI_CLASSES]
    percentages = [c / total * 100 if total > 0 else 0 for c in counts]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    bars = ax1.bar(AAMI_CLASSES, counts, color=["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#9467bd"])
    ax1.set_ylabel("Count (log scale)", fontsize=11)
    ax1.set_yscale("log")
    ax1.set_xlabel("AAMI EC57 Beat Class", fontsize=11)
    ax1.set_title(f"AAMI Beat Class Distribution {title_suffix}", fontsize=13, fontweight="bold")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Annotate counts and percentages on top of bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        if height > 0:
            ax1.annotate(
                f"{count:,}\n({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Visualization] Figure saved to {save_path}")

    if show:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="ECG Visualization Tool")
    parser.add_argument("--record", type=str, default="100", help="Record ID to visualize")
    parser.add_argument("--start", type=float, default=0.0, help="Window start in seconds")
    parser.add_argument("--duration", type=float, default=6.0, help="Window duration in seconds")
    args = parser.parse_args()

    data_dir = get_default_data_dir()
    fig_dir = get_figures_dir()

    print(f"[Visualization] Visualizing record {args.record}...")
    try:
        results = preprocess_record(args.record, data_dir=data_dir)

        # 1. Filter comparison
        comp_path = fig_dir / f"record_{args.record}_preprocessing.png"
        plot_signal_filtering_comparison(
            results, start_sec=args.start, duration_sec=args.duration, save_path=comp_path
        )

        # 2. Morphology plot
        morph_path = fig_dir / f"record_{args.record}_morphologies.png"
        plot_heartbeat_morphologies(
            results["segments"], results["labels"], save_path=morph_path
        )

        # 3. Class distribution plot
        dist_path = fig_dir / f"record_{args.record}_distribution.png"
        plot_class_distribution(
            results["labels"], title_suffix=f"(Record {args.record})", save_path=dist_path
        )

    except FileNotFoundError:
        print(f"[Error] Record {args.record} not found in {data_dir}. Run python -m src.data_loader --download-sample first.")


if __name__ == "__main__":
    main()
