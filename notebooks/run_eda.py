"""
notebooks/run_eda.py
====================
Standalone CLI script to execute the exploratory data analysis pipeline,
inspect metadata, run preprocessing, and save figures to outputs/figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_loader import download_records, get_default_data_dir, load_record, inspect_record
from src.preprocessing import preprocess_record, AAMI_CLASSES, IDX_TO_CLASS
from src.visualization import (
    get_figures_dir,
    plot_signal_filtering_comparison,
    plot_heartbeat_morphologies,
    plot_class_distribution,
)


def main():
    print("=" * 70)
    print("MIT-BIH Arrhythmia Database - Exploratory Data Analysis Pipeline")
    print("=" * 70)

    data_dir = get_default_data_dir()
    fig_dir = get_figures_dir()

    # 1. Download initial sample records
    print("\n[Step 1] Fetching official sample records (100, 101) from PhysioNet...")
    download_records(["100", "101"], output_dir=data_dir)

    # 2. Inspect record metadata
    print("\n[Step 2] Inspecting metadata for Record 100:")
    record, annotation = load_record("100", data_dir=data_dir)
    info = inspect_record(record, annotation)
    print(f"  - Sampling Rate:     {info['sampling_rate_hz']} Hz")
    print(f"  - Duration:          {info['duration_minutes']:.2f} min ({info['total_samples']:,} samples)")
    print(f"  - Available Leads:   {info['leads']}")
    print(f"  - Total Annotations: {info['total_annotations']:,}")
    print(f"  - Raw Symbol Counts: {info['symbol_counts']}")

    # 3. Preprocessing & Segmentation
    print("\n[Step 3] Preprocessing Record 100 (Lead: MLII)...")
    res = preprocess_record("100", data_dir=data_dir, preferred_lead="MLII")
    print(f"  - Extracted {res['num_beats']} heartbeats.")
    print(f"  - Segment matrix shape: {res['segments'].shape} (200 samples per beat = ~555 ms)")

    # 4. Generate & Save Figures
    print("\n[Step 4] Generating exploratory figures...")
    f1 = fig_dir / "eda_record_100_filter_comparison.png"
    plot_signal_filtering_comparison(res, start_sec=10.0, duration_sec=5.0, save_path=f1)

    f2 = fig_dir / "eda_record_100_morphology.png"
    plot_heartbeat_morphologies(res["segments"], res["labels"], save_path=f2)

    f3 = fig_dir / "eda_record_100_class_distribution.png"
    plot_class_distribution(res["labels"], title_suffix="(Record 100)", save_path=f3)

    print("\n" + "=" * 70)
    print("EDA Pipeline Completed Successfully!")
    print(f"Figures saved in: {fig_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
