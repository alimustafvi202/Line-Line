"""
export_for_vercel.py
====================
Exports all MIT-BIH preprocessed signals, R-peaks, model predictions,
and benchmarks into a clean JSON bundle for 100% self-contained Vercel deployment.
"""

import csv
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from src.preprocessing import preprocess_record, IDX_TO_CLASS, AAMI_CLASSES
from src.model import ECGCNN1D
from src.evaluation import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
FRONTEND_DATA_DIR = PROJECT_ROOT / "frontend" / "data"
FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)

RECORDS = ["100", "101", "106", "119", "201", "209"]
MODELS = ["baseline_1d_cnn", "improved_1d_cnn", "multi_arrhythmia_1d_cnn"]

def main():
    print("[Export] Starting export for Vercel deployment...")
    
    # 1. Load trained models
    loaded_models = {}
    for m_name in MODELS:
        m_path = MODELS_DIR / f"{m_name}.pt"
        if m_path.exists():
            model = ECGCNN1D(in_channels=1, num_classes=5)
            model.load_state_dict(torch.load(m_path, map_location=torch.device("cpu")))
            model.eval()
            loaded_models[m_name] = model
            print(f"  - Loaded model: {m_name}")

    # 2. Process records: extract signals, beats, and model predictions
    records_data = {}
    predictions_data = {}
    
    fs = 360
    max_samples = 12 * fs # 12 seconds streaming

    for rec_id in RECORDS:
        print(f"  - Processing record {rec_id}...")
        rec = preprocess_record(rec_id, data_dir=RAW_DIR)
        
        # Signal slice for oscilloscope (round to 3 decimals to save bandwidth)
        sig_slice = [round(float(x), 3) for x in rec["normalized_signal"][:max_samples]]
        
        # R-peaks inside slice
        r_peaks = rec["r_peak_indices"]
        labels = rec["labels"]
        mask = r_peaks < max_samples
        
        beats_meta = []
        for idx, (peak, lbl) in enumerate(zip(r_peaks[mask], labels[mask])):
            beats_meta.append({
                "beat_index": idx,
                "sample": int(peak),
                "time_sec": round(float(peak) / fs, 3),
                "label_idx": int(lbl),
                "class_name": IDX_TO_CLASS[int(lbl)]
            })
            
        records_data[rec_id] = {
            "record": rec_id,
            "lead": rec["lead_name"],
            "sampling_rate": fs,
            "signal": sig_slice,
            "beats": beats_meta,
            "total_record_beats": int(rec["num_beats"])
        }

        # Run inference with all models
        segments = rec["segments"]
        y_true = rec["labels"]
        tensor_x = torch.tensor(segments, dtype=torch.float32).unsqueeze(1)
        
        for m_name, model in loaded_models.items():
            key = f"{rec_id}_{m_name}"
            with torch.no_grad():
                logits = model(tensor_x)
                probs = F.softmax(logits, dim=1).numpy()
                y_pred = np.argmax(probs, axis=1)

            metrics = compute_metrics(y_true, y_pred)

            sample_beats = []
            for i in range(min(25, len(segments))):
                sample_beats.append({
                    "beat_index": i,
                    "true_class": IDX_TO_CLASS[int(y_true[i])],
                    "pred_class": IDX_TO_CLASS[int(y_pred[i])],
                    "confidence": round(float(probs[i, y_pred[i]]), 4),
                    "probabilities": {cls: round(float(probs[i, idx]), 4) for idx, cls in enumerate(AAMI_CLASSES)},
                    "waveform": [round(float(v), 3) for v in segments[i]]
                })

            predictions_data[key] = {
                "record": rec_id,
                "model": m_name,
                "total_beats": len(y_true),
                "accuracy": round(float(metrics["accuracy"]), 4),
                "macro_f1": round(float(metrics["macro_f1"]), 4),
                "per_class": metrics["per_class"],
                "confusion_matrix": metrics["confusion_matrix"],
                "sample_beats": sample_beats
            }

    # 3. Benchmarks from CSV
    benchmarks = []
    csv_path = PROJECT_ROOT / "outputs" / "tables" / "model_comparison.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                benchmarks.append(row)

    bundle = {
        "records": RECORDS,
        "models": MODELS,
        "benchmarks": benchmarks,
        "signals": records_data,
        "predictions": predictions_data
    }

    out_file = FRONTEND_DATA_DIR / "ecgData.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(bundle, f, separators=(",", ":"))

    size_kb = out_file.stat().st_size / 1024
    print(f"[Export] Successfully generated {out_file} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    main()
