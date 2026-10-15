"""
src/data_loader.py
==================
Academic research module for loading and downloading the official
PhysioNet MIT-BIH Arrhythmia Database using the WFDB Python package.

PhysioNet Source: https://physionet.org/content/mitdb/1.0.0/
Standard: 48 two-channel ambulatory ECG recordings, sampled at 360 Hz.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import wfdb


# Official 48 record identifiers in the MIT-BIH Arrhythmia Database
ALL_MITDB_RECORDS: List[str] = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234"
]

# De Chazal et al. (2004) Inter-Patient division
DS1_TRAIN_RECORDS: List[str] = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230"
]

DS2_TEST_RECORDS: List[str] = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234"
]

PACED_EXCLUDED_RECORDS: List[str] = ["102", "104", "107", "217"]


def get_default_data_dir() -> Path:
    """Return the absolute path to data/raw/."""
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def download_records(
    record_names: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    overwrite: bool = False
) -> List[str]:
    """
    Download official MIT-BIH recordings from PhysioNet.

    Parameters
    ----------
    record_names : Optional[List[str]]
        List of record numbers (e.g. ['100', '101']). If None, downloads all 48 records.
    output_dir : Optional[Path]
        Directory where records are stored. Defaults to data/raw.
    overwrite : bool
        If False, skips records that are already present locally.

    Returns
    -------
    List[str]
        List of successfully downloaded/verified record identifiers.
    """
    target_dir = output_dir or get_default_data_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    records_to_fetch = record_names if record_names is not None else ALL_MITDB_RECORDS

    downloaded = []
    print(f"[DataLoader] Target directory: {target_dir}")
    print(f"[DataLoader] Requesting {len(records_to_fetch)} record(s) from PhysioNet mitdb...")

    for rec_id in records_to_fetch:
        # Check if record files (.dat, .hea, .atr) already exist
        dat_file = target_dir / f"{rec_id}.dat"
        hea_file = target_dir / f"{rec_id}.hea"
        atr_file = target_dir / f"{rec_id}.atr"

        if not overwrite and dat_file.exists() and hea_file.exists() and atr_file.exists():
            print(f"  - Record {rec_id}: already exists locally, skipping download.")
            downloaded.append(rec_id)
            continue

        print(f"  - Downloading Record {rec_id}...")
        success = False
        # Try fast PhysioNet Open Data AWS mirror first
        try:
            import urllib.request
            s3_base = "https://physionet-open.s3.amazonaws.com/mitdb/1.0.0"
            for ext in [".hea", ".dat", ".atr"]:
                url = f"{s3_base}/{rec_id}{ext}"
                dest = target_dir / f"{rec_id}{ext}"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as response, open(dest, "wb") as out_file:
                    out_file.write(response.read())
            success = True
            downloaded.append(rec_id)
        except Exception as s3_err:
            # Fallback to wfdb dl_files
            try:
                wfdb.dl_files(
                    db="mitdb",
                    dl_dir=str(target_dir),
                    files=[f"{rec_id}.hea", f"{rec_id}.dat", f"{rec_id}.atr"],
                    keep_subdirs=False
                )
                success = True
                downloaded.append(rec_id)
            except Exception as exc:
                print(f"    [Error] Failed to download record {rec_id}: {exc}")

    print(f"[DataLoader] Download complete. {len(downloaded)}/{len(records_to_fetch)} verified.")
    return downloaded


def load_record(
    record_name: str,
    data_dir: Optional[Path] = None
) -> Tuple[wfdb.Record, wfdb.Annotation]:
    """
    Load a single MIT-BIH record and its reference annotations.

    Parameters
    ----------
    record_name : str
        Record number (e.g. '100').
    data_dir : Optional[Path]
        Directory where data is located.

    Returns
    -------
    Tuple[wfdb.Record, wfdb.Annotation]
        The loaded record object and annotation object.
    """
    path_dir = data_dir or get_default_data_dir()
    rec_path = path_dir / record_name

    if not (path_dir / f"{record_name}.hea").exists():
        raise FileNotFoundError(
            f"Record {record_name} not found in {path_dir}. "
            f"Run python -m src.data_loader --download {record_name} first."
        )

    record = wfdb.rdrecord(str(rec_path))
    annotation = wfdb.rdann(str(rec_path), "atr")
    return record, annotation


def inspect_record(
    record: wfdb.Record,
    annotation: wfdb.Annotation
) -> Dict[str, Any]:
    """
    Extract structured metadata and annotation symbol frequency for a record.

    Parameters
    ----------
    record : wfdb.Record
        The WFDB record.
    annotation : wfdb.Annotation
        The WFDB annotation object.

    Returns
    -------
    Dict[str, Any]
        Dictionary with record statistics, leads, duration, and symbol counts.
    """
    sig_len = record.sig_len
    fs = record.fs
    duration_sec = sig_len / fs
    duration_min = duration_sec / 60.0

    symbols, counts = np.unique(annotation.symbol, return_counts=True)
    symbol_distribution = dict(zip(symbols, [int(c) for c in counts]))

    lead_stats = {}
    for idx, sig_name in enumerate(record.sig_name):
        sig = record.p_signal[:, idx]
        lead_stats[sig_name] = {
            "channel_idx": idx,
            "units": record.units[idx] if record.units else "mV",
            "mean": float(np.nanmean(sig)),
            "std": float(np.nanstd(sig)),
            "min": float(np.nanmin(sig)),
            "max": float(np.nanmax(sig)),
        }

    return {
        "record_name": record.record_name,
        "sampling_rate_hz": fs,
        "total_samples": sig_len,
        "duration_seconds": duration_sec,
        "duration_minutes": duration_min,
        "leads": record.sig_name,
        "total_annotations": len(annotation.sample),
        "symbol_counts": symbol_distribution,
        "lead_statistics": lead_stats
    }


def get_dataset_summary(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Summarize all locally downloaded records in a structured DataFrame.
    """
    target_dir = data_dir or get_default_data_dir()
    available_records = [
        f.stem for f in target_dir.glob("*.hea")
        if (target_dir / f"{f.stem}.dat").exists() and (target_dir / f"{f.stem}.atr").exists()
    ]

    summary_rows = []
    for rec_id in sorted(available_records):
        try:
            rec, ann = load_record(rec_id, target_dir)
            info = inspect_record(rec, ann)
            summary_rows.append({
                "record": info["record_name"],
                "fs": info["sampling_rate_hz"],
                "duration_min": round(info["duration_minutes"], 2),
                "leads": ", ".join(info["leads"]),
                "annotations_total": info["total_annotations"],
                "symbols": str(info["symbol_counts"])
            })
        except Exception as err:
            print(f"[Warning] Could not inspect record {rec_id}: {err}")

    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(
        description="PhysioNet MIT-BIH Arrhythmia Database Loader & Downloader"
    )
    parser.add_argument(
        "--download-sample",
        action="store_true",
        help="Download an initial sample of records (100 and 101) for testing."
    )
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Download the entire official MIT-BIH dataset (48 records)."
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Specific record ID to download or inspect (e.g. 100)."
    )
    parser.add_argument(
        "--inspect",
        type=str,
        default=None,
        help="Inspect a specific local record (e.g. 100) and display metadata."
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Display tabular summary of all locally downloaded records."
    )

    args = parser.parse_args()
    data_dir = get_default_data_dir()

    if args.download_sample:
        print("[Task] Downloading sample records [100, 101]...")
        download_records(["100", "101"], output_dir=data_dir)

    elif args.download_all:
        print("[Task] Downloading full MIT-BIH Arrhythmia Database...")
        download_records(ALL_MITDB_RECORDS, output_dir=data_dir)

    elif args.record:
        print(f"[Task] Downloading record {args.record}...")
        download_records([args.record], output_dir=data_dir)

    if args.inspect:
        rec_id = args.inspect
        print(f"\n[Task] Inspecting Record {rec_id}:")
        rec, ann = load_record(rec_id, data_dir)
        info = inspect_record(rec, ann)
        print(f"  Record: {info['record_name']}")
        print(f"  Sampling frequency: {info['sampling_rate_hz']} Hz")
        print(f"  Duration: {info['duration_minutes']:.2f} minutes ({info['total_samples']} samples)")
        print(f"  Leads: {info['leads']}")
        print(f"  Total Annotations: {info['total_annotations']}")
        print(f"  Annotation Symbol Counts: {info['symbol_counts']}")

    if args.summary:
        print("\n[Task] Generating Dataset Summary Table...")
        df = get_dataset_summary(data_dir)
        if df.empty:
            print("No records found in data/raw/. Run --download-sample first.")
        else:
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()
