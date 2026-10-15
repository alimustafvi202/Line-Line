"""
app.py
======
High-performance local web server and REST API for the
AI ECG Heart Anomaly Research Web Application.

Uses Python's standard library (ThreadingHTTPServer) - requires no extra packages!
Serves the web dashboard and handles live PyTorch neural network inference.
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import ALL_MITDB_RECORDS, get_default_data_dir, load_record
from src.evaluation import compute_metrics
from src.model import ECGCNN1D
from src.preprocessing import AAMI_CLASSES, IDX_TO_CLASS, preprocess_record

WEB_DIR = PROJECT_ROOT / "web"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Cache loaded models in memory
LOADED_MODELS = {}


def get_cached_model(model_name: str) -> ECGCNN1D:
    """Load model with in-memory caching."""
    if model_name in LOADED_MODELS:
        return LOADED_MODELS[model_name]

    model_path = MODELS_DIR / f"{model_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} does not exist.")

    device = torch.device("cpu")
    model = ECGCNN1D(in_channels=1, num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    LOADED_MODELS[model_name] = model
    return model


class ECGAppRequestHandler(SimpleHTTPRequestHandler):
    """Handles web requests and provides JSON REST API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/api/status":
            self.send_json_response({"status": "online", "project": "ECG-Heart-Anomaly-Research"})

        elif path == "/api/records":
            # Find all available records in data/raw
            available = []
            for item in RAW_DATA_DIR.glob("*.hea"):
                rec_id = item.stem
                if (RAW_DATA_DIR / f"{rec_id}.dat").exists() and (RAW_DATA_DIR / f"{rec_id}.atr").exists():
                    available.append(rec_id)
            available.sort()
            self.send_json_response({"records": available})

        elif path == "/api/models":
            # Find all available model checkpoints
            models = [f.stem for f in MODELS_DIR.glob("*.pt")]
            models.sort()
            self.send_json_response({"models": models})

        elif path == "/api/record-signal":
            # Return preprocessed signal and R-peaks for oscilloscope display
            rec_id = query.get("record", ["100"])[0]
            try:
                rec_data = preprocess_record(rec_id, data_dir=RAW_DATA_DIR)
                fs = 360
                # Stream first 12 seconds for fast smooth rendering
                max_samples = 12 * fs
                signal_slice = rec_data["normalized_signal"][:max_samples].tolist()
                
                # Filter R-peaks within this window
                r_peaks = rec_data["r_peak_indices"]
                labels = rec_data["labels"]
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

                self.send_json_response({
                    "record": rec_id,
                    "lead": rec_data["lead_name"],
                    "sampling_rate": fs,
                    "signal": signal_slice,
                    "beats": beats_meta,
                    "total_record_beats": int(rec_data["num_beats"])
                })
            except Exception as exc:
                self.send_error_response(500, f"Error loading record {rec_id}: {str(exc)}")

        elif path == "/api/benchmarks":
            # Read model_comparison.csv
            csv_path = PROJECT_ROOT / "outputs" / "tables" / "model_comparison.csv"
            if csv_path.exists():
                import csv
                rows = []
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        rows.append(r)
                self.send_json_response({"benchmarks": rows})
            else:
                self.send_json_response({"benchmarks": []})

        else:
            # Fallback to standard static file serving (HTML, CSS, JS)
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/predict":
            # Run inference on record with selected model
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                rec_id = data.get("record", "100")
                model_name = data.get("model", "improved_1d_cnn")

                rec_data = preprocess_record(rec_id, data_dir=RAW_DATA_DIR)
                segments = rec_data["segments"]
                y_true = rec_data["labels"]

                model = get_cached_model(model_name)
                tensor_x = torch.tensor(segments, dtype=torch.float32).unsqueeze(1)

                with torch.no_grad():
                    logits = model(tensor_x)
                    probs = F.softmax(logits, dim=1).numpy()
                    y_pred = np.argmax(probs, axis=1)

                metrics = compute_metrics(y_true, y_pred)

                # Return first 20 sample beats with detailed probabilities
                sample_beats = []
                for i in range(min(25, len(segments))):
                    sample_beats.append({
                        "beat_index": i,
                        "true_class": IDX_TO_CLASS[int(y_true[i])],
                        "pred_class": IDX_TO_CLASS[int(y_pred[i])],
                        "confidence": float(probs[i, y_pred[i]]),
                        "probabilities": {cls: float(probs[i, idx]) for idx, cls in enumerate(AAMI_CLASSES)},
                        "waveform": segments[i].tolist()
                    })

                self.send_json_response({
                    "record": rec_id,
                    "model": model_name,
                    "total_beats": len(y_true),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "per_class": metrics["per_class"],
                    "confusion_matrix": metrics["confusion_matrix"],
                    "sample_beats": sample_beats
                })

            except Exception as exc:
                self.send_error_response(500, f"Inference error: {str(exc)}")

        else:
            self.send_error_response(404, "Endpoint not found.")

    def send_json_response(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_error_response(self, status: int, message: str):
        self.send_json_response({"error": message}, status=status)


def run_server(port: int = 5000):
    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, ECGAppRequestHandler)
    print("=" * 70)
    print("AI ECG Heart Anomaly Research Web Application")
    print("=" * 70)
    print(f"Server running at: http://localhost:{port}")
    print(f"Serving web directory: {WEB_DIR}")
    print("Press Ctrl+C to stop.")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        httpd.server_close()


if __name__ == "__main__":
    port = 5000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    run_server(port)
